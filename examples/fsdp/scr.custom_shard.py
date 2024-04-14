#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)

from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

import torch.distributed as dist

from functools import partial

from datetime import timedelta

# ----------------------------------------------------------------------- #
#  Model
# ----------------------------------------------------------------------- #
class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class HeadBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = LinearBlock(input_size=256, output_size=128)
        self.block2 = LinearBlock(input_size=128, output_size=64)
        self.head   = HeadBlock(input_size = 64, output_size = 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x

model = SimpleLinearModel()

# ----------------------------------------------------------------------- #
#  Policy
# ----------------------------------------------------------------------- #
def wrap_block_policy(module: nn.Module) -> bool:
    # Apply FSDP only to block1 and block2
    return isinstance(module, LinearBlock)

custom_auto_wrap_policy = partial(
    lambda_auto_wrap_policy,
    lambda_fn = wrap_block_policy,
)

sharding_strategy = ShardingStrategy.FULL_SHARD


# ----------------------------------------------------------------------- #
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP init
# --- Initialize distributed environment
fsdp_backend = 'nccl'
uses_unique_world_seed = True
uses_fsdp = int(os.environ.get("RANK", -1)) != -1
if uses_fsdp:
    fsdp_rank       = int(os.environ["RANK"      ])
    fsdp_local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = fsdp_backend,
                            rank        = fsdp_rank,
                            world_size  = fsdp_world_size,
                            timeout     = timedelta(seconds=900),
                            init_method = "env://",)
    print(f"RANK:{fsdp_rank},LOCAL_RANK:{fsdp_local_rank},WORLD_SIZE:{fsdp_world_size}")
else:
    fsdp_rank       = 0
    fsdp_local_rank = 0
    fsdp_world_size = 1
    print(f"NO FSDP is used.  RANK:{fsdp_rank},LOCAL_RANK:{fsdp_local_rank},WORLD_SIZE:{fsdp_world_size}")

# --- Set up GPU device
device = f'cuda:{fsdp_local_rank}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = fsdp_rank if uses_unique_world_seed else 0


# ----------------------------------------------------------------------- #
#  FSDP MODEL
# ----------------------------------------------------------------------- #
# Check out the number of parameters before the shard
if fsdp_rank == 0:
    print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")
# -- Wrapping the model in FSDP...
if uses_fsdp:
    # Wrap it up using FSDP...
    model = FSDP(
        model,
        auto_wrap_policy  = custom_auto_wrap_policy,
        sharding_strategy = sharding_strategy,
        limit_all_gathers = True,
        use_orig_params   = True,
        device_id         = device,
    )

    sharded_param_count = sum(p.numel() for p in model.module.parameters())
    print(f"RANK {fsdp_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()
