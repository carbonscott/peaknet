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

echo "Training starts..."
jsrun \
--rs_per_host {{ num_gpus }} \
--tasks_per_rs 1 \
--cpu_per_rs {{ num_cpus }} \
--gpu_per_rs 1 \
--latency_priority gpu-gpu \
--launch_distribution packed \
python {{ trainer }} {{ yaml_config }}

# Kill all running applications (e.g. the ipc server)
jskill all
