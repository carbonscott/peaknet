import yaml

from peaknet.plugins.slac import init_dist_env_on_s3df

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

# -- Debug
# [WARNING] Making it True may throw errors when using float16.
# Invalid gradients are expected to occur during mixed-precision training in
# float16 and anomaly detection will thus report false errors.
# Refer to https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4
torch.autograd.set_detect_anomaly(False)

# ----------------------------------------------------------------------- #
#  MISC FEATURES
# ----------------------------------------------------------------------- #
def signal_handler(signal, frame):
    # Emit Ctrl+C like signal which is then caught by our try/except block
    raise KeyboardInterrupt

# Register the signal handler
signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ----------------------------------------------------------------------- #
#  DIST SETUP
# ----------------------------------------------------------------------- #
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

# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
dtype = torch.bfloat16
B, C, H, W = 100, 1, 512, 512
data = torch.randn(B, C, H, W, dtype = dtype)


# ----------------------------------------------------------------------- #
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP policy
# --- Mixed precision
dist_dtype = 'bfloat16'
mixed_precision = MixedPrecision(
    param_dtype  = torch.bfloat16,
    reduce_dtype = torch.bfloat16,
    buffer_dtype = torch.bfloat16,
)

# --- Sharding strategy
sharding_strategy = ShardingStrategy.FULL_SHARD
## sharding_strategy = ShardingStrategy.NO_SHARD

# --- Wrapping strategy
# ---- Use built-in transformer wrap policy
def create_auto_wrap_policy(target_layer_classes, min_num_params):
    def should_wrap_module(module):

        num_params_in_module = sum(p.numel() for p in module.parameters())
        large_enough = num_params_in_module >= min_num_params

        return (
            isinstance(module, target_layer_classes)
            and
            large_enough
        )

    return partial(lambda_auto_wrap_policy, lambda_fn=should_wrap_module)

wrap_layer_cls= (
    ## ConvNextV2Embeddings,
    ConvNextV2Stage,
    ## ConvNextV2Layer, # Need to experiment with it
    ## BiFPN,
    ## SegLateralLayer,
)
wrap_min_num_params = 2000384  # pow of 2 friendly

auto_wrap_policy = create_auto_wrap_policy(wrap_layer_cls, wrap_min_num_params)

# --- Activation checkpointing
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu  = False,
    checkpoint_impl = CheckpointImpl.NO_REENTRANT,
)

# --- Backward prefetch policy
backward_prefetch = BackwardPrefetch.BACKWARD_PRE


# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
model_yaml_config = yaml.safe_load("""
      num_channels     : 1
      patch_size       : 4
      num_stages       : 4
      hidden_sizes     : [96, 192, 384, 768]
      depths           : [3, 3, 9, 3]
      hidden_act       : "gelu"
      initializer_range: 0.02
      layer_norm_eps   : !!float 1e-12
      drop_path_rate   : 0.0
      image_size       : 512
      out_features     : ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
      out_indices      : null
""")

model_config = ConvNextV2Config(**model_yaml_config)
model = ConvNextV2Backbone(model_config)

# -- Mixed precision
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = mixed_precision_dtype)

# --- GradScaler
# If enabled = False scaler is a no-op
scaler_func = ShardedGradScaler if uses_dist else torch.cuda.amp.GradScaler
scaler = scaler_func(enabled=(dist_dtype == 'float16'))

if uses_dist:
    # Convert BatchNorm to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using FSDP
    model = FSDP(
        model,
        auto_wrap_policy  = auto_wrap_policy,
        mixed_precision   = mixed_precision,
        backward_prefetch = backward_prefetch,
        forward_prefetch  = True,
        sharding_strategy = sharding_strategy,
        limit_all_gathers = True,
        use_orig_params   = True,
        device_id         = device,
    )

    ## logger.debug(f"[RANK {dist_rank}] {report_fsdp_details(model, dist_rank, dist_world_size)}")

    sharded_param_count = sum(p.numel() for p in model.parameters())  # .module will return the raw model view when use_orig_params = True
                                                                      # making it effectively reporting the sharded param count.  Removing
                                                                      # .module makes it more consistent regardles of use_orig_params.
    print(f"RANK {dist_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    print(model)

    dist.barrier()


# -- Optional grad sync off (to allow grad accumulation)
grad_sync_context = lambda enables_sync: nullcontext() if enables_sync or not uses_dist else model.no_sync()

# -- Apply activation checkpointing
ac_layer = ConvNextV2Stage
if ac_layer is not None:
    check_fn = lambda submodule: isinstance(submodule, ac_layer)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn = non_reentrant_wrapper,
        check_fn              = check_fn
    )


# Forward pass
iterations = 10
for _ in range(iterations):
    output = model(data)
