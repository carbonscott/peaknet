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

from mpi4py import MPI

def init_dist_env_on_s3df():
    # Use mpi4py to get rank and size information
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # Calculate local rank based on the available GPUs
    mpi_local_rank = mpi_rank % torch.cuda.device_count()

    # Are we using multiple ranks?
    uses_dist = mpi_size > 1

    if uses_dist:
        MAIN_RANK = 0

        # MPI environment variables detected (e.g., Summit)
        os.environ["WORLD_SIZE"] = str(mpi_size)
        os.environ["RANK"]       = str(mpi_rank)
        os.environ["LOCAL_RANK"] = str(mpi_local_rank)

        # Set the default master address and port, prioritizing definition in the job script
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")

        master_addr = os.getenv("MASTER_ADDR", None)
        if master_addr is None:
            # Try to determine the master address and broadcast it to every rank
            master_addr = socket.gethostbyname(socket.gethostname()) if mpi_rank == MAIN_RANK else None
            master_addr = mpi_comm.bcast(master_addr, root = MAIN_RANK)
            os.environ["MASTER_ADDR"] = master_addr
        else:
            os.environ["MASTER_ADDR"] = "127.0.0.1"

        print(f"Environment setup for distributed computation: WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ['RANK']}, LOCAL_RANK={os.environ['LOCAL_RANK']}, MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
