#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#!SBATCH --constraint gpu         # Use GPU 
#!SBATCH --partition=ampere
#SBATCH --partition=milano
#!SBATCH --qos=debug              # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#!SBATCH --time 00:29:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#!SBATCH --qos=regular        # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#SBATCH --time 00:10:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --exclusive
#SBATCH --job-name=dataloading
#!SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80

cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj.peaknet

echo "sbatch /sdf/data/lcls/ds/prj/prjcwang31/results/proj.peaknet"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Function to start gunicorn server
start_process() {
    echo "Starting server on $(hostname)..."
    python server.ipc.py &

    sleep 10

    echo "Running client on $(hostname)..."
    python client.ipc.py
}

export -f start_process

# Launch the server on each node
srun bash -c start_process
