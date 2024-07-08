#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#SBATCH --partition=ampere
#SBATCH --time {{ walltime }}          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --exclusive
#SBATCH --job-name={{ job }}
#SBATCH --gres=gpu:{{ num_gpus_per_node }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{ num_cpus_per_node }}

# Set up the Huggingface's cache directory
export TRANSFORMERS_CACHE={{ transformers_cache }}

export OMP_NUM_THREADS={{ OMP_NUM_THREADS }}
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

# Set up a meta checkpoint file
export PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
export PREEMPT_METADATA_PATH="$PREEMPT_ROOT/{{ job }}.dat"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun                        \
torchrun                    \
--nnodes {{ num_nodes }}                  \
--nproc_per_node {{ num_gpus_per_node }}          \
--rdzv_id $RANDOM           \
--rdzv_backend c10d         \
--rdzv_endpoint $head_node_ip:29500 \
{{ trainer }} {{ yaml_config }}
