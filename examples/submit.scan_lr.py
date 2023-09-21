#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import subprocess

from configurator import Configurator


# Read the base configuration...
fl_yaml = "base_config.yaml"
with open(fl_yaml, 'r') as fh:
    config_dict = yaml.safe_load(fh)
CONFIG = Configurator.from_dict(config_dict)

# Set up export directory...
# ...Base
drc_base    = 'experiments'
job_basename = f"lr"

# ...Checkpoint
CONFIG.CHKPT.DIRECTORY = os.path.join(drc_base, 'chkpts')
os.makedirs(CONFIG.CHKPT.DIRECTORY, exist_ok = True)

# ...Checkpoint
CONFIG.LOGGING.DIRECTORY = os.path.join(drc_base, 'logs')
os.makedirs(CONFIG.LOGGING.DIRECTORY, exist_ok = True)

# ...YAML
drc_yaml = os.path.join(drc_base, 'yaml')
os.makedirs(drc_yaml, exist_ok = True)

# ...Slurm
drc_slurm = os.path.join(drc_base, 'slurm')
os.makedirs(drc_slurm, exist_ok = True)


# Specify the configuration to scan and the range...
lr_scan_range = [2e-3, 1e-3, 0.5e-3]

# Write the slurm job template...
cwd = os.getcwd()

def custom_slurm_content(job_name, cwd, trainer, path_output_slurm, path_output_yaml):
    return  (
        f"#!/bin/bash\n"
        f"#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid\n"
        f"#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid\n"
        f"#SBATCH --account lcls_g         # Check it in your Iris portal: https://iris.nersc.gov\n"
        f"#SBATCH --constraint gpu         # Use GPU \n"
        f"#!SBATCH --qos=debug              # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss\n"
        f"#!SBATCH --time 00:29:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/\n"
        f"#SBATCH --qos=regular            # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss\n"
        f"#SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --gres=gpu:4\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=6\n"
        f"\n"
        f"cd {cwd}\n"
        f"\n"
        f"echo \"sbatch {path_output_slurm}\n"
        f"\n"
        f"nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) ) \n"
        f"nodes_array=($nodes)\n"
        f"head_node=${{nodes_array[0]}}\n"
        f"head_node_ip=$(srun --nodes=1 --ntasks=1 -w \"$head_node\" hostname --ip-address)\n"
        f"\n"
        f"echo Node IP: $head_node_ip\n"
        f"export LOGLEVEL=INFO\n"
        f"\n"
        f"torchrun                    \\\n"
        f"--nnodes 1                  \\\n"
        f"--nproc_per_node 4          \\\n"
        f"--rdzv_id $RANDOM           \\\n"
        f"--rdzv_backend c10d         \\\n"
        f"--rdzv_endpoint localhost:0 \\\n"
        f"{trainer} {path_output_yaml}"
    )

trainer = f"train.res_bifpn_net.ddp.py"

for enum_idx, lr in enumerate(lr_scan_range):
    job_name = f"{job_basename}.{enum_idx:02d}"
    # Get a new scan value...
    CONFIG.OPTIM.LR = lr
    output_config = CONFIG.to_dict()

    # Write to a YAML file...
    fl_output_yaml = f"{job_name}.yaml"
    path_output_yaml = os.path.join(drc_yaml, fl_output_yaml)
    with open(path_output_yaml, 'w') as fh:
        yaml.dump(output_config, fh)

    # Write a slurm script...
    fl_output_slurm = f"{job_name}.sbatch"
    path_output_slurm = os.path.join(drc_slurm, fl_output_slurm)
    slurm_output_content = custom_slurm_content(job_name, cwd, trainer, path_output_slurm, path_output_yaml)
    with open(path_output_slurm, 'w') as fh:
        fh.write(slurm_output_content)
