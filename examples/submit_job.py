#!/usr/bin/env python
# -*- coding: utf-8 -*-

from jinja2 import Environment, FileSystemLoader

import os
import sys
import shutil
import yaml
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Generate experiment scripts from templates.")
parser.add_argument("--job_basename" , type = str, help = "Value for job name.")
parser.add_argument("--path_train"   , type = str, help = "Path to the dataset csv (train).")
parser.add_argument("--path_validate", type = str, help = "Path to the dataset csv (validate).")
parser.add_argument("--model_name"   , type = str, help = "Name of the backbone")
parser.add_argument("--num_nodes"    , type = int, help = "Number of computer nodes.")
parser.add_argument("--num_gpus"     , type = int, help = "Number of GPUs.")
parser.add_argument("--num_workers"  , type = int, help = "Number of workers to load data.")
parser.add_argument("--batch_size"   , type = int, help = "Batch size.")
parser.add_argument("--trainer"      , type = str, help = "The training script.")
args = parser.parse_args()

job_basename  = args.job_basename
path_train    = args.path_train
path_validate = args.path_validate
model_name    = args.model_name
num_nodes     = args.num_nodes
num_gpus      = args.num_gpus
num_workers   = args.num_workers
batch_size    = args.batch_size
trainer       = args.trainer

cpu_per_task = (num_workers * num_gpus) + 2

# Set up the environment and the loader
env = Environment(loader=FileSystemLoader('.'))
drc_template = 'template'
bash_template = env.get_template(os.path.join(drc_template, 'slurm.multi_node.s3df.sh' if num_nodes > 1 else 'slurm.single_node.s3df.sh'))
yaml_template = env.get_template(os.path.join(drc_template, 'config.yaml'))

# Render the yaml script
yaml_content = yaml_template.render(filename_prefix = job_basename,
                                    path_train      = path_train,
                                    path_validate   = path_validate,
                                    model_name      = model_name,
                                    num_nodes       = num_nodes,
                                    num_gpus        = num_gpus,
                                    batch_size      = batch_size,
                                    num_workers     = num_workers,)

config_dict = yaml.safe_load(yaml_content)

# Set up export directory...
# ...Base
drc_base = 'experiments'

# Copy the training and validation csv under experiments folder...
drc_datasets = os.path.join(drc_base, 'datasets')
os.makedirs(drc_datasets, exist_ok = True)
shutil.copy2(path_train   , drc_datasets)
shutil.copy2(path_validate, drc_datasets)

# ...Checkpoint
checkpoint_params = config_dict.get("checkpoint")
directory         = checkpoint_params.get("directory")
os.makedirs(directory, exist_ok = True)

# ...Logging
logging_params = config_dict.get("logging")
directory      = logging_params.get("directory")
os.makedirs(directory, exist_ok = True)

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
misc_params = config_dict.get("misc")
num_nodes = misc_params.get("num_nodes")
num_gpus  = misc_params.get("num_gpus")

for enum_idx, num_bifpn_block in enumerate(scan_range):
    job_name = f"{job_basename}.{enum_idx:02d}"

    # Write to a YAML file...
    yaml_content = yaml_template.render(filename_prefix = job_name,
                                        path_train      = path_train,
                                        path_validate   = path_validate,
                                        model_name      = model_name,
                                        num_nodes       = num_nodes,
                                        num_gpus        = num_gpus,
                                        batch_size      = batch_size,
                                        num_workers     = num_workers,)

    fl_output_yaml = f"{job_name}.yaml"
    path_output_yaml = os.path.join(drc_yaml, fl_output_yaml)
    with open(path_output_yaml, 'w') as fh:
        header = '# python ' + ' '.join(sys.argv)
        fh.write(header)
        fh.write('\n')

        fh.write(yaml_content)
        ## yaml.dump(config_dict, fh)

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
