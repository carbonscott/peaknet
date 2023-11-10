#!/usr/bin/env python
# -*- coding: utf-8 -*-

from jinja2 import Environment, FileSystemLoader

import os
import yaml
import argparse
import subprocess

from peaknet.configurator import Configurator

parser = argparse.ArgumentParser(description="Generate experiment scripts from templates.")
parser.add_argument("--job_basename"           , type = str, help = "Value for job name.")
parser.add_argument("--path_pretrain_chkpt", type = str, help = "Path to a pretrained weight.")
parser.add_argument("--path_dataset"       , type = str, help = "Path to the dataset (CXI).")
parser.add_argument("--sample_size"        , type = int, help = "Sample size (Train + Validation).")
parser.add_argument("--num_nodes"          , type = int, help = "Number of computer nodes.")
parser.add_argument("--num_gpus"           , type = int, help = "Number of GPUs.")
parser.add_argument("--num_workers"        , type = int, help = "Number of workers to load data.")
parser.add_argument("--trainer"            , type = str, help = "The training script.")
args = parser.parse_args()

job_basename        = args.job_basename
path_pretrain_chkpt = args.path_pretrain_chkpt
path_dataset        = args.path_dataset
sample_size         = args.sample_size
num_nodes           = args.num_nodes
num_gpus            = args.num_gpus
num_workers         = args.num_workers
trainer             = args.trainer

cpu_per_task = (num_workers * num_gpus) + 2

# Set up the environment and the loader
env = Environment(loader=FileSystemLoader('.'))
drc_template = 'template'
bash_template = env.get_template(os.path.join(drc_template, 'slurm.multi_node.sh' if num_nodes > 1 else 'slurm.single_node.sh'))
yaml_template = env.get_template(os.path.join(drc_template, 'config.yaml'))

# Render the yaml script
yaml_content = yaml_template.render(filename_prefix     = job_basename,
                                    path_pretrain_chkpt = path_pretrain_chkpt,
                                    path_dataset        = path_dataset,
                                    sample_size         = sample_size,
                                    num_nodes           = num_nodes,
                                    num_gpus            = num_gpus,
                                    num_workers         = num_workers,)

config_dict = yaml.safe_load(yaml_content)
CONFIG      = Configurator.from_dict(config_dict)

# Set up export directory...
# ...Base
drc_base = 'experiments'

# ...Checkpoint
os.makedirs(CONFIG.CHKPT.DIRECTORY, exist_ok = True)

# ...Logging
os.makedirs(CONFIG.LOGGING.DIRECTORY, exist_ok = True)

# ...YAML
drc_yaml = os.path.join(drc_base, 'yaml')
os.makedirs(drc_yaml, exist_ok = True)

# ...Slurm
drc_slurm = os.path.join(drc_base, 'sbatch')
os.makedirs(drc_slurm, exist_ok = True)

# Specify the configuration to scan and the range...
scan_range = [0, ]

# Write the slurm job template...
cwd = os.getcwd()

# Get num of gpus...
num_nodes = CONFIG.MISC.NUM_NODES
num_gpus  = CONFIG.MISC.NUM_GPUS

for enum_idx, num_bifpn_block in enumerate(scan_range):
    job_name = f"{job_basename}.{enum_idx:02d}"

    ## # ___/ Get a new scan value \___
    ## CONFIG.MODEL.BIFPN.NUM_BLOCKS = num_bifpn_block

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
    slurm_output_content = bash_template.render(job_name          = job_name,
                                                num_nodes         = num_nodes,
                                                num_gpus          = num_gpus,
                                                cwd               = cwd,
                                                trainer           = trainer,
                                                cpu_per_task      = cpu_per_task,
                                                path_output_slurm = path_output_slurm,
                                                path_output_yaml  = path_output_yaml)
    with open(path_output_slurm, 'w') as fh:
        fh.write(slurm_output_content)
