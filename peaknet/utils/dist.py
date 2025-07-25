# ====================================================================================================
# Distributed Environment Initialization Process:
# - mpi4py Usage:
#   - mpi4py is used to obtain rank and size information for distributed processing.
#   - It provides a standardized way to get this information across different systems.
#
# - Environment Variable Setup:
#   - We set environment variables like WORLD_SIZE, RANK, and LOCAL_RANK.
#   - This adheres to the torchrun convention, which uses these environment variables.
#   - PyTorch's distributed module can then use these variables for its setup.
#
# ====================================================================================================

import os
import socket
import torch
import torch.distributed as dist
from datetime import timedelta
from omegaconf import OmegaConf
import sys

def init_dist_env_with_mpi(device_per_node=None):
    """Initialize distributed environment using MPI."""
    try:
        from mpi4py import MPI
    except ImportError:
        raise RuntimeError("mpi4py is not found!!!")

    # Use mpi4py to get rank and size information
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # Calculate local rank based on the available GPUs
    mpi_local_rank = mpi_rank % device_per_node

    # Are we using multiple ranks?
    uses_dist = mpi_size > 1

    # Set basic environment variables
    os.environ["WORLD_SIZE"] = str(mpi_size) if uses_dist else "1"
    os.environ["RANK"] = str(mpi_rank) if uses_dist else "0"
    os.environ["LOCAL_RANK"] = str(mpi_local_rank) if uses_dist else "0"
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")

    # Only handle MASTER_ADDR if it's not already set in environment
    if "MASTER_ADDR" not in os.environ:
        if uses_dist:
            MAIN_RANK = 0
            # Try to determine the master address and broadcast it to every rank
            master_addr = socket.gethostbyname(socket.gethostname()) if mpi_rank == MAIN_RANK else None
            master_addr = mpi_comm.bcast(master_addr, root=MAIN_RANK)
            os.environ["MASTER_ADDR"] = master_addr
        else:
            # Single rank case
            os.environ["MASTER_ADDR"] = "127.0.0.1"

    print(f"Environment setup for distributed computation: "
          f"WORLD_SIZE={os.environ['WORLD_SIZE']}, "
          f"RANK={os.environ['RANK']}, "
          f"LOCAL_RANK={os.environ['LOCAL_RANK']}, "
          f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
          f"MASTER_PORT={os.environ['MASTER_PORT']}")

def init_dist_env_with_srun():
    uses_dist = int(os.environ.get('SLURM_NTASKS',1)) > 1
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS'] if uses_dist else "1"
    os.environ['RANK'] = os.environ['SLURM_PROCID'] if uses_dist else "0"
    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID'] if uses_dist else "0"
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")
    if "MASTER_ADDR" not in os.environ:
        if uses_dist:
            raise RuntimeError(
                "Error: MASTER_ADDR environment variable is not set.\n"
                "Please set it before launching with srun using:\n"
                "    export MASTER_ADDR=$(hostname)"
            )
        else:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
    print(f"Environment setup for distributed computation: "
          f"WORLD_SIZE={os.environ['WORLD_SIZE']}, "
          f"RANK={os.environ['RANK']}, "
          f"LOCAL_RANK={os.environ['LOCAL_RANK']}, "
          f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
          f"MASTER_PORT={os.environ['MASTER_PORT']}")

def dist_setup(cpu_only, device_per_node=1, dist_backend='nccl'):
    # -- DIST init
    # --- S3DF/SLAC specific env (adapted from OLCF version)
    # torchrun doesn't work well on some HPC systems. 
    is_rank_setup = int(os.environ.get("RANK", -1)) != -1
    if device_per_node is None:
        device_per_node = torch.cuda.device_count()
    if not is_rank_setup:
        is_srun_used = int(os.environ.get('SLURM_NTASKS',-1)) != -1
        if is_srun_used:
            init_dist_env_with_srun()
        elif int(os.environ.get("WORLD_SIZE", 1)) > 1:
            # Only try MPI if WORLD_SIZE > 1 (distributed training expected)
            init_dist_env_with_mpi(device_per_node)
        # else: single GPU case, no need to initialize distributed environment

    # --- Set up GPU device first to determine device_id for init_process_group
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    uses_dist  = world_size > 1

    gpu_idx = local_rank % device_per_node    # local_rank is node-centric, whereas torch.cuda.device_count() is resource-centeric (on LSF)
    device = f'cuda:{gpu_idx}' if not cpu_only and torch.cuda.is_available() else 'cpu'

    # --- Initialize distributed environment with proper device_id
    if uses_dist:
        # Set device before init_process_group to avoid warnings
        if device != 'cpu': 
            torch.cuda.set_device(device)
            device_id = torch.device(device)  # Convert to torch.device object
        else:
            device_id = None

        # Initialize with device_id to prevent warnings about unknown devices
        init_kwargs = {
            "backend": dist_backend,
            "rank": rank,
            "world_size": world_size,
            "timeout": timedelta(seconds=1800),
            "init_method": "env://",
        }

        # Add device_id for NCCL backend with CUDA devices
        if dist_backend == 'nccl' and device_id is not None:
            init_kwargs["device_id"] = device_id

        dist.init_process_group(**init_kwargs)
        print(f"RANK:{rank},LOCAL_RANK:{local_rank},WORLD_SIZE:{world_size},DEVICE:{device}")
    else:
        if device != 'cpu': 
            torch.cuda.set_device(device)
        print(f"NO distributed environment is required.  RANK:{rank},LOCAL_RANK:{local_rank},WORLD_SIZE:{world_size},DEVICE:{device}")
    return rank, local_rank, world_size, uses_dist, device
