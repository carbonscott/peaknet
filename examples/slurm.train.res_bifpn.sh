#!/bin/bash
#SBATCH --job-name=PF            # Job name
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls_g         # Check it in your Iris portal: https://iris.nersc.gov
#SBATCH --constraint gpu         # Use GPU
#!SBATCH --qos=debug             # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#!SBATCH --time 00:29:00         # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --qos=regular            # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#!SBATCH --gres=gpu:1            # GPU resources requested: 1x A100.
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node 1      # Numebr of tasks (e.g. train.py)
#SBATCH --cpus-per-task 6        # Number of CPUs per task
#SBATCH --gpus-per-task 1        # Number of GPUs per task
#SBATCH --mem=180GB              # Total CPU memory requested

############################################################
# More examples about running gpu jobs on Perlmutter (NERSC)
# - https://docs.nersc.gov/systems/perlmutter/running-jobs/
# - https://my.nersc.gov/script_generator.php (might be outdated)
############################################################

# Assume your conda environment has been activated.
python train.res_bifpn_net.py
