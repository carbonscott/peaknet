#!/bin/bash

#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls_g         # Check it in your Iris portal: https://iris.nersc.gov
#SBATCH --constraint gpu         # Use GPU
#!SBATCH --qos=debug              # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#!SBATCH --time 00:29:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --qos=regular            # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --job-name=MULTIN_PF
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

torchrun                    \
--nnodes 1                  \
--nproc_per_node 3          \
--rdzv_id $RANDOM           \
--rdzv_backend c10d         \
--rdzv_endpoint localhost:0 \
train.res_bifpn_net.ddp.py


## torchrun                    \
## --nnodes 2                  \
## --nproc_per_node 2          \
## --rdzv_id $RANDOM           \
## --rdzv_backend c10d         \
## --rdzv_endpoint $head_node_ip:29500 \
## --maskter_addr=${head_node_ip} \
## --maskter_port=1234 \
## train.res_bifpn_net.ddp.py
