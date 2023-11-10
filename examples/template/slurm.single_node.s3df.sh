#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#!SBATCH --constraint gpu         # Use GPU 
#SBATCH --partition=ampere
#!SBATCH --qos=debug              # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#!SBATCH --time 00:29:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#!SBATCH --qos=regular        # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --job-name={{ job_name }}
#SBATCH --gres=gpu:{{ num_gpus }}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{ cpu_per_task|default(11) }}

cd {{ cwd }}

echo "sbatch {{ path_output_slurm }}"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun                        \
torchrun                    \
--nnodes 1                  \
--nproc_per_node {{ num_gpus }}          \
--rdzv_id $RANDOM           \
--rdzv_backend c10d         \
--rdzv_endpoint localhost:0 \
{{ trainer }} {{ path_output_yaml }}
