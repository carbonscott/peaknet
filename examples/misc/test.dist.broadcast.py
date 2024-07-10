import yaml

from peaknet.plugins.slac import init_dist_env_on_s3df

import pickle
import os
import signal

from functools  import partial
from contextlib import nullcontext
from datetime   import timedelta

from peaknet.utils_fsdp import (
    MemoryMaximizer,
    verify_bfloat_support,
    FullStateDictCheckpointConfig,
    FullStateDictCheckpoint,
    broadcast_dict,
    init_logger,
)

from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import (
    ConvNextV2Backbone,
    ConvNextV2Embeddings,
    ConvNextV2Stage,
    ConvNextV2Layer,
)

# -- Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim

# -- Fully Sharded Data Parallel (FSDP)
# --- Main
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)

# --- Policy wrapper
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    lambda_auto_wrap_policy,
)
from packaging import version

# --- Scaler for float16
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# --- Activation checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

# --- Distributed library
import torch.distributed as dist

def broadcast_dict(obj, src=0, device = 'cpu'):
    rank = dist.get_rank()
    if rank == src:
        # Serialize the dictionary...
        buffer = pickle.dumps(obj)
        ## tensor = torch.ByteTensor(list(buffer), device = device)
        tensor = torch.tensor(list(buffer), dtype=torch.uint8, device=device)

        # Communicate about the size of the underlying data...
        tensor_size = torch.tensor([len(buffer)], dtype=torch.long, device = device)
    else:
        # Prepare to receive the size of the underlying data...
        tensor_size = torch.tensor([0], dtype=torch.long, device = device)

    # Broadcast the size of the tensor to all processes...
    dist.broadcast(tensor_size, src)

    # Prepare to receive data...
    if rank != src:
        tensor = torch.empty((tensor_size.item(),), dtype=torch.uint8, device = device)

    # Broadcast the data...
    dist.broadcast(tensor, src)

    if rank != src:
        # Deserialize the tensor back into a dictionary...
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)

    return obj

# -- DIST init
# --- OLCF specific env
torchrun_exists = int(os.environ.get("RANK", -1)) != -1
if not torchrun_exists: init_dist_env_on_s3df()

# --- Initialize distributed environment
dist_backend = 'nccl'
uses_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
if uses_dist:
    dist_rank       = int(os.environ["RANK"      ])
    dist_local_rank = int(os.environ["LOCAL_RANK"])
    dist_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = dist_backend,
                            rank        = dist_rank,
                            world_size  = dist_world_size,
                            timeout     = timedelta(seconds = 1800),
                            init_method = "env://",)
    print(f"RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")
else:
    dist_rank       = 0
    dist_local_rank = 0
    dist_world_size = 1
    print(f"NO distributed environment is required.  RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")

# --- Set up GPU device
gpu_idx = dist_local_rank % torch.cuda.device_count()    # dist_local_rank is node-centric, whereas torch.cuda.device_count() is resource-centeric (on LSF)
cpu_only = False
device = f'cuda:{gpu_idx}' if not cpu_only and torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)


def broadcast_tensor(tensor, rank):
    """Broadcast a tensor from rank 0 to all other ranks."""

    # Step 1: Rank 0 sends the shape
    if rank == 0:
        shape = torch.tensor(tensor.shape, dtype=torch.int64)
    else:
        shape = torch.empty((len(tensor.shape),), dtype=torch.int64)

    # Broadcast the shape from rank 0 to all other ranks
    dist.broadcast(shape, src=0)

    # Step 2: All ranks (except 0) create an empty tensor with the received shape
    if rank != 0:
        tensor = torch.empty(tuple(shape.tolist()), dtype=tensor.dtype)

    # Step 3: Broadcast the actual tensor data from rank 0 to all other ranks
    dist.broadcast(tensor, src=0)

    return tensor

## data_dict = None
## 
## if dist_rank == 0:
##     dtype = torch.bfloat16
##     B, C, H, W = 100, 1, 512, 512
##     data_dict = {'data' : torch.tensor((B, C, H, W), device = device)}
##     print(f"[RANK {dist_rank}] BEFORE data_dict = {data_dict}, pointer = {data_dict.get('data').data_ptr()}")
## 
## data_dict = broadcast_dict(data_dict, src = 0, device = device)
## ## data = data.to(device)
## 
## print(f"[RANK {dist_rank}] data_dict = {data_dict}, pointer = {data_dict.get('data').data_ptr()}")
## 
## dummy_data_cuda_0 = torch.randn(data_dict.get('data').tolist(), device = 'cuda:0')
## print(f"[RANK {dist_rank}] {dummy_data_cuda_0.device}, pointer = {dummy_data_cuda_0.data_ptr()}")


## obj_list = [None, ]
## if dist_rank == 0:
##     dtype = torch.bfloat16
##     B, C, H, W = 100, 1, 512, 512
##     data = torch.randn((B, C, H, W), device = device)
##     obj_list = [data, ]
## 
## dist.broadcast_object_list(obj_list, device = device)
## 
## data_fetch = obj_list[0]
## print(f"[RANK {dist_rank}] data_fetch.shape = {data_fetch.shape}, device = {data_fetch.device}, pointer = {data_fetch.data_ptr()}")

B, C, H, W = 100, 1, 512, 512
data = torch.empty(B, C, H, W, device = device)
if dist_rank == 0:
    dtype = torch.bfloat16
    data = torch.randn((B, C, H, W), device = device)

dist.broadcast(data, src = 0)
print(f"[RANK {dist_rank}] data.shape = {data.shape}, device = {data.device}, pointer = {data.data_ptr()}")
