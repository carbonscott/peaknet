# Monkey patch the `build_metadata` method
# Refer to
# - Discussion: https://github.com/pytorch/pytorch/pull/108915
# - Diff: https://github.com/pytorch/pytorch/pull/108915/files
# - Codes: https://github.com/b-chu/pytorch/blob/3ccc8f3357447f83ce56dd9d7618e520e4a11e97/torch/distributed/_shard/sharding_spec/chunk_sharding_spec.py
import torch
import torch.distributed as dist
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.checkpoint import optimizer
from pkg_resources import packaging

from .utils_patch_torch import patched_build_metadata, patched_shard, patched_load_sharded_optimizer_state_dict

def patch_torch():
    torch_version = torch.__version__
    torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
    if packaging.version.parse(torch_version) <= packaging.version.parse("2.0.1"):
        ChunkShardingSpec.build_metadata = patched_build_metadata
        print(f"[RANK {dist.get_rank()}] ChunkShardingSpec.build_metadata has been patched...")

        ## ChunkShardingSpec.shard = patched_shard
        ## print(f"[RANK {dist.get_rank()}] ChunkShardingSpec.shard has been patched...")

        ## optimizer.load_sharded_optimizer_state_dict = patched_load_sharded_optimizer_state_dict
        ## print(f"[RANK {dist.get_rank()}] optimizer.load_sharded_optimizer_state_dict has been patched...")
