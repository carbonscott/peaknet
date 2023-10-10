#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import subprocess

from peaknet.configurator import Configurator


# Read the base configuration...
fl_yaml = "base_config.yaml"
with open(fl_yaml, 'r') as fh:
    config_dict = yaml.safe_load(fh)
CONFIG = Configurator.from_dict(config_dict)

# Set up export directory...
# ...Base
drc_base    = 'experiments'
job_basename = f"debug2.syntax"

# ...Checkpoint
CONFIG.CHKPT.DIRECTORY = os.path.join(drc_base, 'chkpts')
os.makedirs(CONFIG.CHKPT.DIRECTORY, exist_ok = True)

# ...Logging
CONFIG.LOGGING.DIRECTORY = os.path.join(drc_base, 'logs')
os.makedirs(CONFIG.LOGGING.DIRECTORY, exist_ok = True)

# ...YAML
drc_yaml = os.path.join(drc_base, 'yaml')
os.makedirs(drc_yaml, exist_ok = True)

# ...Slurm
drc_slurm = os.path.join(drc_base, 'slurm')
os.makedirs(drc_slurm, exist_ok = True)

# Specify if unique world seed should be used...
CONFIG.DDP.USES_UNIQUE_WORLD_SEED = True

# Specify learning rate...
CONFIG.OPTIM.LR = 3e-4    # See notes in experiments/notes/scan_lr.md

# Specify the configuration to scan and the range...
scan_range = [3, ]

# Write the slurm job template...
cwd = os.getcwd()

# Get num of gpus...
num_nodes = CONFIG.MISC.NUM_NODES
num_gpus  = CONFIG.MISC.NUM_GPUS

def custom_slurm_content(job_name, num_nodes, num_gpus, cwd, trainer, path_output_slurm, path_output_yaml):
    return  (
        f"#!/bin/bash\n"
        f"#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid\n"
        f"#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid\n"
        f"#SBATCH --account lcls_g         # Check it in your Iris portal: https://iris.nersc.gov\n"
        f"#SBATCH --constraint gpu         # Use GPU \n"
        f"#SBATCH --qos=debug              # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss\n"
        f"#SBATCH --time 00:29:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/\n"
        f"#!SBATCH --qos=regular            # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss\n"
        f"#!SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --gres=gpu:{num_gpus}\n"
        f"#SBATCH --nodes={num_nodes}\n"
        f"#SBATCH --ntasks-per-node=1\n"
        f"#SBATCH --cpus-per-task=11\n"
        f"#!SBATCH --gpus-per-task=1\n"
        f"\n"
        f"cd {cwd}\n"
        f"\n"
        f"echo \"sbatch {path_output_slurm}\"\n"
        f"\n"
        f"nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )\n"
        f"head_node=${{nodes[0]}}\n"
        f"head_node_ip=$(srun --nodes=1 --ntasks=1 -w \"$head_node\" hostname --ip-address)\n"
        f"head_node_ip=$(echo \"$head_node_ip\" | awk '{{print $1}}')\n"
        f"\n"
        f"echo Node IP: $head_node_ip\n"
        f"export LOGLEVEL=INFO\n"
        f"\n"
        f"# Multi-nodes...\n"
        f"rdzv_id=$RANDOM\n"
        f"worker_num=$(($SLURM_JOB_NUM_NODES - 1)) \n"
        f"for ((  i=0; i<=$worker_num; i++ ))\n"
        f"do\n"
        f"  node_i=${{nodes[$i]}}\n"
        f"  srun --nodelist=$node_i             \\\n"
        f"  torchrun                            \\\n"
        f"  --nnodes {num_nodes}                \\\n"
        f"  --nproc_per_node {num_gpus}         \\\n"
        f"  --node-rank $i                      \\\n"
        f"  --rdzv_id $rdzv_id                  \\\n"
        f"  --rdzv_backend c10d                 \\\n"
        f"  --rdzv_endpoint $head_node_ip:29500 \\\n"
        f"  {trainer} {path_output_yaml} &\n"
        f"  pids[${{i}}]=$!\n"
        f"  echo \"Training started on node $i, $node_i, $head_node_ip.\"\n"
        f"done\n"
        f"\n"
        f"# Wait for completion\n"
        f"for pid in ${{pids[*]}}; do\n"
        f"    wait $pid\n"
        f"done\n"
        f"\n"
        f"date\n"
    )

trainer = f"train.res_bifpn_net.ddp.py"

for enum_idx, num_bifpn_block in enumerate(scan_range):
    job_name = f"{job_basename}.{enum_idx:02d}"
    # Get a new scan value...
    CONFIG.MODEL.BIFPN.NUM_BLOCKS = num_bifpn_block

    # Specify chkpt and log filename...
    CONFIG.CHKPT.FILENAME_PREFIX   = job_name
    CONFIG.LOGGING.FILENAME_PREFIX = job_name

    # Export config...
    output_config = CONFIG.to_dict()

    # Write to a YAML file...
    fl_output_yaml = f"{job_name}.yaml"
    path_output_yaml = os.path.join(drc_yaml, fl_output_yaml)
    with open(path_output_yaml, 'w') as fh:
        yaml.dump(output_config, fh)

    # Write a slurm script...
    fl_output_slurm = f"{job_name}.sbatch"
    path_output_slurm = os.path.join(drc_slurm, fl_output_slurm)
    slurm_output_content = custom_slurm_content(job_name, num_nodes, num_gpus, cwd, trainer, path_output_slurm, path_output_yaml)
    with open(path_output_slurm, 'w') as fh:
        fh.write(slurm_output_content)
