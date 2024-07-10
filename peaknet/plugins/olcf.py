import os
import torch.distributed as dist

def init_dist_env_on_summit():
    uses_dist = int(os.environ.get("OMPI_COMM_WORLD_SIZE", -1)) != -1

    if uses_dist:
        # MPI environment variables detected (e.g., Summit)
        os.environ["WORLD_SIZE"] = os.getenv("OMPI_COMM_WORLD_SIZE")
        os.environ["RANK"]       = os.getenv("OMPI_COMM_WORLD_RANK")
        os.environ["LOCAL_RANK"] = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")

        # Address and port
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "8888")
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "127.0.0.1")
        if os.getenv("LSB_HOSTS") is not None:
            ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
            ## The following is Summit specific
            os.environ["MASTER_ADDR"] = os.environ["LSB_HOSTS"].split()[1]
        elif os.getenv("LSB_MCPU_HOSTS") is not None:
            os.environ["MASTER_ADDR"] = os.environ["LSB_MCPU_HOSTS"].split()[2]

        print(f"Environment setup for distributed computation: WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ['RANK']}, LOCAL_RANK={os.environ['LOCAL_RANK']}")
