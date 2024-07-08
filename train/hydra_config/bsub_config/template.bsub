#!/bin/bash
#BSUB -o lsf/%J.log
#BSUB -e lsf/%J.err
#BSUB -q {{ qos }}
#BSUB -W {{ walltime }}
#BSUB -P LRN044
#BSUB -J {{ job }}
#BSUB -nnodes {{ num_nodes }}

# Set up the Huggingface's cache directory
export TRANSFORMERS_CACHE={{ transformers_cache }}

export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/

export OMP_NUM_THREADS={{ OMP_NUM_THREADS }}
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

# Set up a meta checkpoint file
export PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
export PREEMPT_METADATA_PATH="$PREEMPT_ROOT/{{ job }}.dat"

# Check if a checkpoint exists and resume from it
if [ -f $PREEMPT_METADATA_PATH ]; then
    echo "Resuming from preemptive checkpoint..."
    python -c "import yaml, os; data = yaml.safe_load(open('{{ yaml_config }}')); data['checkpoint']['path_chkpt_prev'] = open(os.getenv('PREEMPT_METADATA_PATH')).read().strip(); yaml.safe_dump(data, open('{{ yaml_config }}', 'w'))"
fi

# Fetch all nodes and output a whole string of concatenated host nodes
# $LSB_MCPU_HOSTS gives something like "batch02 1 a09n03 42 a10n04 42".
# I need just "a09n03 a10n04" to set up a head node.
nodelist=$(echo $LSB_MCPU_HOSTS | awk '{for (i=3; i<=NF; i+=2) print $i}' | sort | uniq)    # "a09n03 a10n04"
read -r -a nodes <<< "$nodelist"
head_node=${nodes[0]}
head_node_ip=$(ssh "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

echo "Starting server..."
jsrun \
--tasks_per_rs 1 \
--cpu_per_rs {{ num_cpus_for_server }} \
--gpu_per_rs 0 \
--rs_per_host 1 \
--latency_priority cpu-cpu \
--launch_distribution packed \
python server.ipc.py --num_workers {{ ipc_workers }} &

sleep 10

echo "Running client..."
jsrun \
--rs_per_host {{ num_gpus_for_client }} \
--tasks_per_rs 1 \
--cpu_per_rs {{ num_cpus_for_client }} \
--gpu_per_rs 1 \
--latency_priority gpu-gpu \
--launch_distribution packed \
python {{ trainer }} {{ yaml_config }}

# Kill all running applications (e.g. the ipc server)
jskill all
