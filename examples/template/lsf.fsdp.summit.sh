#!/bin/bash
#BSUB -o lsf/%J.log
#BSUB -e lsf/%J.err
#!BSUB -q batch
#!BSUB -q batch-hm
#BSUB -q debug
#BSUB -W 2:00
#BSUB -P LRN044
#BSUB -J {{ job_name }}
#!BSUB -alloc_flags gpudefault
#BSUB -nnodes {{ num_nodes }}

export OMP_NUM_THREADS=1
## export http_proxy=http://proxy.ccs.ornl.gov:3128/
## export https_proxy=https://proxy.ccs.ornl.gov:3128/

export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

cd {{ cwd }}

echo "sbatch {{ path_output_slurm }}"

# Fetch all nodes and output a whole string of concatenated host nodes
# $LSB_MCPU_HOSTS gives something like "batch02 1 a09n03 42 a10n04 42".
# I need just "a09n03 a10n04" to set up a head node.
nodelist=$(echo $LSB_MCPU_HOSTS | awk '{for (i=3; i<=NF; i+=2) print $i}' | sort | uniq)    # "a09n03 a10n04"
read -r -a nodes <<< "$nodelist"
head_node=${nodes[0]}
head_node_ip=$(ssh "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo "Node IP: $head_node_ip ($head_node)"
export LOGLEVEL=INFO

jsrun                        \
-n {{ num_nodes }} \
--tasks_per_rs 1 \
--cpu_per_rs {{ cpu_per_task }} \
--gpu_per_rs {{ num_gpus }} \
--rs_per_host 1 \
--latency_priority gpu-cpu \
--launch_distribution packed \
torchrun                    \
--nnodes {{ num_nodes }}                  \
--nproc_per_node {{ num_gpus }}          \
--rdzv_id $RANDOM           \
--rdzv_backend c10d         \
--rdzv_conf timeout=900     \
--rdzv_endpoint $head_node_ip:32760 \
{{ trainer }} {{ path_output_yaml }}
