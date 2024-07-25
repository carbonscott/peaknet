#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- S3DF specific imports
from peaknet.plugins.slac import init_dist_env_on_s3df
## from peaknet.plugins.olcf import init_dist_env_on_summit

# -- Basic imports
import os
import yaml
import tqdm
import signal
import argparse
import inspect
import logging
import traceback
import time

from functools  import partial
from contextlib import nullcontext
from datetime   import timedelta

# -- peaknet specific imports
# --- Dataset
from peaknet.datasets.segmented_safetensor_dataset import (
    SegmentedPeakNetDatasetConfig,
    SegmentedPeakNetDataset,
)
from peaknet.tensor_transforms import (
    Pad,
    DownscaleLocalMean,
    RandomPatch,
    RandomRotate,
    RandomShift,
    InstanceNorm,
)

# --- Model
from peaknet.modeling.convnextv2_bifpn_net import (
    SegLateralLayer,
    SegHeadConfig,
    PeakNetConfig,
    PeakNet,
)
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import (
    ConvNextV2PreTrainedModel,
    ConvNextV2Backbone,
    ConvNextV2Embeddings,
    ConvNextV2Stage,
    ConvNextV2Layer,
)
from peaknet.modeling.bifpn_config import (
    BiFPNConfig,
    BiFPNBlockConfig,
    BNConfig,
    FusionConfig,
)
from peaknet.modeling.bifpn import BiFPNBlock, BiFPN

# --- Loss
from peaknet.criterion import CategoricalFocalLoss

# --- Others
from peaknet.utils.seed        import set_seed
from peaknet.utils.misc        import is_action_due
from peaknet.utils.checkpoint  import Checkpoint
from peaknet.lr_scheduler      import CosineLRScheduler
from peaknet.perf              import Timer
from peaknet.utils_fsdp import (
    MemoryMaximizer,
    verify_bfloat_support,
    FullStateDictCheckpoint,
    ShardedStateDictCheckpoint,
    init_logger,
)
from peaknet.utils.monitor import (
    ActivationMonitor,
    monitor_param_update_metrics,
)

# -- Imports for monitoring training dynamics
from transformers.activations import ACT2CLS

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

# -- Reporting specific imports
import colorama
colorama.init(autoreset = True)

# -- Get the logger
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  COMMAND LINE INTERFACE
# ----------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description = "Load training configuration from a YAML file to a dictionary.")
parser.add_argument("yaml_file", help="Path to the YAML file")
args = parser.parse_args()

# ----------------------------------------------------------------------- #
#  CONFIGURATION
# ----------------------------------------------------------------------- #
# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# -- Checkpoint
chkpt_config                    = config.get("checkpoint")
dir_root_chkpt                  = chkpt_config.get("directory")
fl_chkpt_prefix                 = chkpt_config.get("prefix")
path_chkpt_prev                 = chkpt_config.get("path_chkpt_prev")
chkpt_saving_iterations         = chkpt_config.get("chkpt_saving_iterations")
preempt_chkpt_saving_iterations = chkpt_config.get("preempt_chkpt_saving_iterations")
state_dict_type                 = chkpt_config.get("state_dict_type")

# -- Dataset
dataset_config       = config.get("dataset")
path_dataset_train   = dataset_config.get("path_train")
path_dataset_eval    = dataset_config.get("path_eval")
drop_last_in_sampler = dataset_config.get("drop_last_in_sampler")
drop_last_in_loader  = dataset_config.get("drop_last_in_loader")
batch_size           = dataset_config.get("batch_size")
seg_size             = dataset_config.get("seg_size")
num_workers          = dataset_config.get("num_workers")
pin_memory           = dataset_config.get("pin_memory")
prefetch_factor      = dataset_config.get("prefetch_factor")
debug_dataloading    = dataset_config.get("debug")
cache_size           = dataset_config.get("cache_size")
transforms_config    = dataset_config.get("transforms")
num_patch            = transforms_config.get("num_patch")
size_patch           = transforms_config.get("size_patch")
frac_shift_max       = transforms_config.get("frac_shift_max")
angle_max            = transforms_config.get("angle_max")
var_size_patch       = transforms_config.get("var_size_patch")
downscale_factors    = transforms_config.get("downscale_factors")
H_pad                = transforms_config.get("H_pad")
W_pad                = transforms_config.get("W_pad")
patch_size           = transforms_config.get("patch_size")
stride               = transforms_config.get("stride")
detector_norm_params = transforms_config.get("norm")
sampling_fraction    = transforms_config.get("sampling_fraction", None)
set_transforms       = transforms_config.get("set")
uses_pad             = set_transforms.get('pad')
uses_downscale       = set_transforms.get('downscale')
uses_random_patch    = set_transforms.get('random_patch')
uses_random_rotate   = set_transforms.get('random_rotate')
uses_random_shift    = set_transforms.get('random_shift')
uses_instance_norm   = set_transforms.get('instance_norm')


# -- Model
model_params              = config.get("model")
from_scratch              = model_params.get("from_scratch")
backbone_config           = model_params.get("backbone")
hf_model_config           = backbone_config.get("hf_config")
bifpn_params              = model_params.get("bifpn")
bifpn_num_blocks          = bifpn_params.get("num_blocks")
bifpn_block_params        = bifpn_params.get("block")
bifpn_block_bn_params     = bifpn_block_params.get("bn")
bifpn_block_fusion_params = bifpn_block_params.get("fusion")
seghead_params            = model_params.get("seg_head")
seghead_num_classes       = seghead_params.get("num_classes")

# -- Loss
loss_config      = config.get("loss")
grad_accum_steps = max(int(loss_config.get("grad_accum_steps")), 1)
focal_config     = loss_config.get("focal")
focal_alpha      = focal_config.get("alpha")
focal_gamma      = focal_config.get("gamma")

# -- Optimizer
optim_config = config.get("optim")
lr           = float(optim_config.get("lr"))
weight_decay = float(optim_config.get("weight_decay"))
adam_beta1   = float(optim_config.get("beta1"))
adam_beta2   = float(optim_config.get("beta2"))
adam_fused   = float(optim_config.get("fused"))
grad_clip    = float(optim_config.get("grad_clip"))

# -- Scheduler
lr_scheduler_config         = config.get("lr_scheduler")
patience                    = lr_scheduler_config.get("patience")
warmup_iterations           = lr_scheduler_config.get("warmup_iterations")
total_iterations            = lr_scheduler_config.get("total_iterations")
min_lr                      = float(lr_scheduler_config.get("min_lr"))
scheduler_update_iterations = lr_scheduler_config.get("scheduler_update_iterations")

# -- Distributed envs
dist_config            = config.get("dist")
dist_backend           = dist_config.get("backend")
uses_unique_world_seed = dist_config.get("uses_unique_world_seed")
dist_dtype             = dist_config.get("dtype")

# -- Logging
logging_config = config.get("logging")
drc_log        = logging_config.get("directory")
fl_log_prefix  = logging_config.get("prefix")
log_level      = logging_config.get("level")

# -- Misc
misc_config        = config.get("misc")
max_epochs         = misc_config.get("max_epochs")
max_eval_iter      = misc_config.get("max_eval_iter")
max_eval_retry     = misc_config.get("max_eval_retry")
compiles_model     = misc_config.get("compiles_model")
data_dump_on       = misc_config.get("data_dump_on", False)
cpu_only           = misc_config.get("cpu_only", False)
peak_flops_per_sec = misc_config.get("peak_flops_per_sec")
monitors_dynamics  = misc_config.get("monitors_dynamics")
sharding_stage     = misc_config.get("sharding_stage")

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
## if not torchrun_exists: init_dist_env_on_summit()

# --- Initialize distributed environment
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
device = f'cuda:{gpu_idx}' if not cpu_only and torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = dist_rank if uses_unique_world_seed else 0

# --- Set up performance utility
memmax = MemoryMaximizer() if dist_local_rank == 0 else None


# ----------------------------------------------------------------------- #
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP policy
# --- Sharding strategy
sharding_strategy = dict(
    zero3 = ShardingStrategy.FULL_SHARD,
    zero2 = ShardingStrategy.SHARD_GRAD_OP,
    zero0 = ShardingStrategy.NO_SHARD,
)[sharding_stage]

# --- Wrapping strategy
# ---- Use built-in transformer wrap policy
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        ## ConvNextV2Embeddings,
        ConvNextV2Stage,
        ## ConvNextV2Layer, # Need to experiment with it
        ## BiFPN,
        ## SegLateralLayer,
    },
)

## def create_auto_wrap_policy(target_layer_classes, min_num_params):
##     def should_wrap_module(module):
## 
##         num_params_in_module = sum(p.numel() for p in module.parameters())
##         large_enough = num_params_in_module >= min_num_params
## 
##         return (
##             isinstance(module, target_layer_classes)
##             and
##             large_enough
##         )
## 
##     return partial(lambda_auto_wrap_policy, lambda_fn = should_wrap_module)
## 
## wrap_layer_cls= (
##     ## ConvNextV2Embeddings,
##     ConvNextV2Stage,
##     ## ConvNextV2Layer, # Need to experiment with it
##     ## BiFPN,
##     ## SegLateralLayer,
## )
## wrap_min_num_params = 2000384  # pow of 2 friendly
## 
## auto_wrap_policy = create_auto_wrap_policy(wrap_layer_cls, wrap_min_num_params)


# --- Activation checkpointing
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    ## offload_to_cpu  = False,
    checkpoint_impl = CheckpointImpl.NO_REENTRANT,
)

# --- Backward prefetch policy
backward_prefetch = BackwardPrefetch.BACKWARD_PRE


# ----------------------------------------------------------------------- #
#  TF32 support
# ----------------------------------------------------------------------- #
# Ampere architecture (capability_major = 8) is required.
if device != 'cpu':
    capability_major, capability_minor = torch.cuda.get_device_capability(device)
    if capability_major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if dist_rank == 0:
            logger.info("[RANK {dist_rank}] TF32 enabled on matmul and cuDNN operations.")


# ----------------------------------------------------------------------- #
#  LOGGING
# ----------------------------------------------------------------------- #
# Fetch the current timestamp...
timestamp = init_logger(
    uses_dist,
    dist_rank,
    device,
    fl_prefix = fl_log_prefix,
    drc_log = drc_log,
    level = log_level,
)

if dist_rank == 0:
    # Convert dictionary to yaml formatted string...
    config_yaml = yaml.dump(config)

    # Log the config...
    logger.info(config_yaml)

# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
logger.debug(f'[RANK {dist_rank}] Configuring dataset...')
# -- Seeding
base_seed  = 0
world_seed = base_seed + seed_offset
set_seed(world_seed)

# -- Set up transformation
class NoTransform:
    def __call__(self, x, **kwargs):
        return x

pre_transforms = (
    Pad(H_pad, W_pad) if uses_pad else NoTransform(),
    DownscaleLocalMean(factors = downscale_factors) if uses_downscale else NoTransform(),
)

transforms = (
    ## Norm(detector_norm_params),

    ## Pad(H_pad, W_pad)                               if uses_pad           else NoTransform(),
    ## DownscaleLocalMean(factors = downscale_factors) if uses_downscale     else NoTransform(),
    RandomPatch(
        num_patch    = num_patch,
        H_patch      = size_patch,
        W_patch      = size_patch,
        var_H_patch  = var_size_patch,
        var_W_patch  = var_size_patch,
        returns_mask = False,
    ) if uses_random_patch  else NoTransform(),
    RandomRotate(angle_max) if uses_random_rotate else NoTransform(),
    RandomShift(
        frac_y_shift_max = frac_shift_max,
        frac_x_shift_max = frac_shift_max,
    ) if uses_random_shift  else NoTransform(),
    InstanceNorm() if uses_instance_norm else NoTransform(),

    ## Patchify(patch_size, stride),
    ## BatchSampler(sampling_fraction),
)

# -- Set up training set
dataset_train_config = SegmentedPeakNetDatasetConfig(
    path_csv        = path_dataset_train,
    seg_size        = seg_size,
    transforms      = pre_transforms,
    buffer_size     = 1,
    dist_rank       = dist_rank,
    dist_world_size = dist_world_size,
    device          = device,
    dtype           = None,
    perfs_runtime   = False,
)
dataset_train = SegmentedPeakNetDataset(dataset_train_config)

# -- Set up eval set
# --- For training loss
dataset_eval_train = SegmentedPeakNetDataset(dataset_train_config)

# --- For val loss
dataset_eval_val_config = SegmentedPeakNetDatasetConfig(
    path_csv        = path_dataset_eval,
    seg_size        = seg_size,
    transforms      = pre_transforms,
    buffer_size     = 1,
    dist_rank       = dist_rank,
    dist_world_size = dist_world_size,
    device          = device,
    dtype           = None,
    perfs_runtime   = False,
)
dataset_eval_val = SegmentedPeakNetDataset(dataset_eval_val_config)

# -- Custom collate to merge patch and batch dimension using concatenation
## custom_collate = lambda batch: torch.cat(batch, dim = 0)  # batch of [N, C, H, W] -> [B * N, C, H, W]
## def custom_collate(batch):
##     batch_filtered = [x for x in batch if x is not None]
##     return torch.cat(batch_filtered, dim = 0) if len(batch_filtered) else None
custom_collate = None

# ----------------------------------------------------------------------- #
#  CHECKPOINT PRE FSDP
# ----------------------------------------------------------------------- #
checkpoint_func = {
    "full"    : FullStateDictCheckpoint,
    "sharded" : ShardedStateDictCheckpoint,
}[state_dict_type] if uses_dist else Checkpoint
checkpointer = checkpoint_func()
from_resume = path_chkpt_prev is not None

# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
logger.debug(f'[RANK {dist_rank}] Configuring model...')
# -- Monkey patch the _init_weights
## # Account for the (pre)activation spread due to the accumulation of residual paths
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Normalize the init std by the number of residual paths
        std  = self.config.initializer_range
        ## std *= (2 * self.config.num_hidden_layers)**-0.5  # 1/sqrt(num_residual_layers), cf: GPT-2 paper

        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
ConvNextV2PreTrainedModel._init_weights = _init_weights

# --- Backbone
backbone_config = ConvNextV2Config(**hf_model_config)

# --- BiFPN
bifpn_block_params["bn"]     = BNConfig        (**bifpn_block_bn_params)
bifpn_block_params["fusion"] = FusionConfig    (**bifpn_block_fusion_params)
bifpn_params["block"]        = BiFPNBlockConfig(**bifpn_block_params)
bifpn_config                 = BiFPNConfig     (**bifpn_params)

# --- Seg head
seghead_config = SegHeadConfig(**seghead_params)

# --- PeakNet
peaknet_config = PeakNetConfig(
    backbone = backbone_config,
    bifpn    = bifpn_config,
    seg_head = seghead_config,
)

# -- Config the model
model = PeakNet(peaknet_config)
if not uses_dist: model.to(device)

# !! Make all params trainable, a workaround for pytorch 2.0.1
torch_version = torch.__version__
torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
if version.parse(torch_version) <= version.parse("2.0.1"):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

if dist_rank == 0:
    logger.debug(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

if from_resume:
    if isinstance(checkpointer, checkpoint_func):
        checkpointer.pre_fsdp_load(dist_rank, model, path_chkpt_prev)

# -- Mixed precision
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]
mixed_precision = MixedPrecision(
    param_dtype  = mixed_precision_dtype,
    reduce_dtype = mixed_precision_dtype,
    buffer_dtype = mixed_precision_dtype,
)

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = mixed_precision_dtype)

# --- GradScaler
# If enabled = False scaler is a no-op
scaler_func = ShardedGradScaler if uses_dist else torch.cuda.amp.GradScaler
scaler = scaler_func(enabled=(dist_dtype == 'float16'))

# -- Compile the model
if compiles_model:
    logger.debug("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# -- Wrapping the model in FSDP
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

    sharded_param_count = sum(p.numel() for p in model.parameters())  # .module will return the raw model view when use_orig_params = True
                                                                      # making it effectively reporting the sharded param count.  Removing
                                                                      # .module makes it more consistent regardles of use_orig_params.
    logger.debug(f"RANK {dist_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()

# -- Optional grad sync off (to allow grad accumulation)
grad_sync_context = lambda enables_sync: nullcontext() if enables_sync or not uses_dist else model.no_sync()

# -- Apply activation checkpointing
## def enable_activation_checkpointing(target_layer_classes, min_num_params):
##     def should_wrap_module(module):
## 
##         num_params_in_module = sum(p.numel() for p in module.parameters())
##         large_enough = num_params_in_module >= min_num_params
## 
##         return (
##             isinstance(module, target_layer_classes)
##             and
##             large_enough
##         )
## 
##     return should_wrap_module

## apply_activation_checkpointing(
##     model,
##     checkpoint_wrapper_fn = non_reentrant_wrapper,
##     check_fn              = enable_activation_checkpointing(wrap_layer_cls, wrap_min_num_params),
## )
ac_layer = ConvNextV2Stage
if ac_layer is not None:
    check_fn = lambda submodule: isinstance(submodule, ac_layer)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn = non_reentrant_wrapper,
        check_fn              = check_fn
    )

if dist_rank == 0:
    logger.debug(f"Current timestamp: {timestamp}")

# ----------------------------------------------------------------------- #
#  CRITERION (LOSS)
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring criterion...')
criterion = CategoricalFocalLoss(alpha       = focal_alpha,
                                 gamma       = focal_gamma,
                                 num_classes = seghead_num_classes,)


# ----------------------------------------------------------------------- #
#  OPTIMIZER AND SCHEDULER
# ----------------------------------------------------------------------- #
logger.debug(f'[RANK {dist_rank}] Configuring optimizer...')
param_iter = model.parameters()
optim_arg_dict = dict(
    lr           = lr,
    weight_decay = weight_decay,
    betas        = (adam_beta1, adam_beta2),
)
if 'fused' in inspect.signature(optim.AdamW).parameters:
    optim_arg_dict['fused'] = adam_fused
optimizer = optim.AdamW(param_iter, **optim_arg_dict)
scheduler = CosineLRScheduler(optimizer         = optimizer,
                              warmup_iterations = warmup_iterations,
                              total_iterations  = total_iterations,
                              min_lr            = min_lr)


# ----------------------------------------------------------------------- #
#  CHECKPOINT POST FSDP
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring model, optim, scheduler, training state checkpoint...')
# -- Set init training state dict
loss_min = float('inf')
iter_state = dict(
    epoch     = 0,
    seg       = 0,
    start_idx = dataset_train.start_idx,
    end_idx   = dataset_train.end_idx,
    loss_min  = loss_min,
)

# -- Optional resumption
last_epoch = 0
last_seg   = -1
if from_resume:
    if isinstance(checkpointer, checkpoint_func):
        # Optimizer, scheduler are loaded
        checkpointer.post_fsdp_load(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt_prev)


        # Training state
        last_epoch = iter_state.get("epoch")
        last_seg   = iter_state.get("seg")
        loss_min   = iter_state.get("loss_min")

        logger.info(f"Loading from checkpoint -- {path_chkpt_prev}.")
        logger.info(f"PREV - last_epoch {last_epoch}, last_seg {iter_state.get('start_idx')}-{iter_state.get('end_idx')}, loss_min = {loss_min}")


# ----------------------------------------------------------------------- #
#  Monitoring training dynamics
# ----------------------------------------------------------------------- #
if monitors_dynamics:
    modules_to_monitor = (ACT2CLS[model.backbone.config.hidden_act], )
    act_monitor = ActivationMonitor(model, modules_to_monitor)
    act_monitor.add_hooks()

# ----------------------------------------------------------------------- #
#  HELPER
# ----------------------------------------------------------------------- #
@torch.no_grad()
def estimate_loss(
    dataloader,
    model,
    criterion,
    autocast_context,
    max_iter              = None,
    desc                  = '',
    device                = 'cpu',
    dummy_input_shape     = None,
    mixed_precision_dtype = torch.float32,
    transforms            = None,
    **kwargs
):
    ''' Estimate loss.
        The dataloader should be wrapped with Dataloader class or
        DistributedSampler class, best with shuffle being true.  The shuffle
        takes place before batching.
    '''
    # -- Setup
    uses_dist       = kwargs.get('uses_dist')
    dist_rank       = kwargs.get('dist_rank')
    dist_world_size = kwargs.get('dist_world_size')

    if dist_rank == 0:
        logger.debug(f"[RANK {dist_rank}] - EVAL Entering")
    model.eval()

    # !!!!!!!!!!!!!!!
    # !! Data dump !!
    # !!!!!!!!!!!!!!!
    if dist_rank == 0 and data_dump_on:
        dir_data_dump = "data_dump"
        os.makedirs(dir_data_dump, exist_ok = True)

        fl_log_prefix = kwargs.get('fl_log_prefix')
        epoch         = kwargs.get('epoch')
        seg           = kwargs.get('seg')

    # -- Eval iterations
    # Set default number of iterations
    if max_iter is None:
        max_iter = len(dataloader)

    losses      = torch.zeros(len(dataloader), device = device)
    num_samples = torch.zeros(len(dataloader), device = device)
    proc_masks  = torch.zeros(len(dataloader), device = device)  # A mask to track the process
    none_mask   = torch.zeros(len(dataloader), device = device)  # Mask for None batches
    for enum_idx, batch_data in tqdm.tqdm(enumerate(dataloader), total = max_iter, desc = f'[RANK {dist_rank}] Eval{desc}'):    # (B, C, H, W)
        # Sample at most max_iter batches
        if enum_idx >= max_iter: break

        if dist_rank == 0:
            logger.debug(f"[RANK {dist_rank}] EVAL - Pre fetching mini_batch {enum_idx}")

        # Create dummy data for a None batch
        # FIXME: Better data cleaning will eliminate None batch
        if batch_data is None:
            logger.debug(f"[RANK {dist_rank}] Found None batch at batch idx {enum_idx}.  Creating a dummy input!!!")
            batch_input  = torch.zeros(dummy_input_shape, dtype = mixed_precision_dtype)
            batch_target = torch.zeros(dummy_input_shape, dtype = mixed_precision_dtype)
            none_mask[enum_idx] = 1
            batch_data = (batch_input, batch_target)

        # -- Optional batch data transforms on GPUs to improve mfu
        # Concat data to perform the identical transform on input and target
        batch_data = torch.cat(batch_data, dim = 0)    # (2*B, C, H, W)
        batch_data = batch_data.to(device, non_blocking = True, dtype = mixed_precision_dtype)

        # Optional transform
        if transforms is not None:
            for enum_idx, trans in enumerate(transforms):
                batch_data = trans(batch_data)

        # Unpack vars
        current_batch_size = batch_data.size(0) // 2
        batch_input  = batch_data[                  :current_batch_size]
        batch_target = batch_data[current_batch_size:                  ]

        # Optionally binarize the label
        if transforms is not None:
            batch_target = batch_target > 0.5

        if dist_rank == 0:
            logger.debug(f"[RANK {dist_rank}] EVAL - Post fetching")

        # -- Forward pass
        with autocast_context:
            if dist_rank == 0:
                logger.debug(f"[RANK {dist_rank}] EVAL - Forwarding")

            batch_output = model(batch_input)

            # !!!!!!!!!!!!!!!
            # !! Data dump !!
            # !!!!!!!!!!!!!!!
            if dist_rank == 0 and data_dump_on:
                mini_batch = enum_idx

                data_dump = {
                    "batch_input"  : batch_input,
                    "batch_target" : batch_target,
                    "batch_output" : batch_output,
                }
                path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.epoch{epoch}_seg{seg}_minib{mini_batch}.fwd.pt')
                torch.save(data_dump, path_data_dump)

            if dist_rank == 0:
                logger.debug(f"[RANK {dist_rank}] EVAL - Loss")
            loss = criterion(batch_output, batch_target)
            loss = loss.mean()

        # !!!!!!!!!!!!!!!
        # !! Data dump !!
        # !!!!!!!!!!!!!!!
        if dist_rank == 0 and data_dump_on:
            mini_batch = enum_idx

            data_dump = {
                "batch_input"  : batch_input,
                "batch_target" : batch_target,
                "batch_output" : batch_output,
                "loss"         : loss,
            }
            path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.epoch{epoch}_seg{seg}_minib{mini_batch}.loop.pt')
            torch.save(data_dump, path_data_dump)

        losses     [enum_idx] = loss
        num_samples[enum_idx] = len(batch_input)
        proc_masks [enum_idx] = 1

    # -- Handle nan
    # Obtain the nan mask
    non_nan_mask = ~torch.isnan(losses)

    # Get the actual mask of values that are from the processing loop and non nan
    masks = torch.logical_and(proc_masks>0, non_nan_mask)
    masks = torch.logical_and(masks, none_mask==0)  # Keep not-None elements

    # -- Mean loss over eval iterations
    local_valid_losses = losses[masks].to(torch.float32)
    local_losses_mean  = local_valid_losses.mean()  # torch.isnan(torch.tensor([]).mean()) -> True

    # -- Mean loss over ranks
    # Survey the occurence of nan across ranks
    world_nan_counter = torch.tensor(0, dtype = torch.int, device = device)
    local_nan_masks = torch.isnan(local_losses_mean)
    if local_nan_masks.any().item():
        logger.error(f"[RANK {dist_rank}] EVAL ERROR: NaN encountered!!!")
        world_nan_counter += 1
        local_losses_mean  = 0.0    # Contribute to nothing in the reduced sum
    if uses_dist: dist.all_reduce(world_nan_counter, op = dist.ReduceOp.SUM)

    # Scale the local loss for the final reduced sum
    local_losses_mean /= (dist_world_size - world_nan_counter + 1e-6)

    # Calculate reduced sum as the final mean loss
    world_losses_mean  = torch.zeros_like(local_losses_mean, dtype = torch.float32, device = device)
    world_losses_mean += local_losses_mean.to(torch.float32)
    if uses_dist: dist.all_reduce(world_losses_mean, op = dist.ReduceOp.SUM)

    # !!!!!!!!!!!!!!!
    # !! Data dump !!
    # !!!!!!!!!!!!!!!
    if dist_rank == 0 and data_dump_on:
        data_dump = {
            "losses"            : losses,
            "proc_masks"        : proc_masks,
            "non_nan_mask"      : non_nan_mask,
            "masks"             : masks,
            "local_valid_losses": local_valid_losses,
            "local_losses_mean" : local_losses_mean,
            "world_losses_mean" : world_losses_mean,
        }
        path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.epoch{epoch}_seg{seg}.end.pt')
        torch.save(data_dump, path_data_dump)

    model.train()

    return world_losses_mean


def is_last_batch(batch_idx, num_batches):
    return batch_idx + 1 == num_batches


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_conv_flops(
        H_input, W_input, C_input, C_output,
        H_kernel, W_kernel, H_stride, W_stride,
        H_padding, W_padding, count_multiply_add_as=1
    ):
    """
    Estimate the number of FLOPs for a convolutional layer.

    This function implements the following logic:
    - Calculate the number of times the kernel is applied (n_H * n_W).
    - Calculate the FLOPs for a single 2D kernel application.
    - Multiply by the number of input and output channels.

    The calculation takes into account:
    - Input dimensions (height, width, channels)
    - Output channels
    - Kernel dimensions
    - Stride
    - Padding

    Count each multiply-add as this many FLOPs (default is 1)
    """
    # Calculate flops for a single 2D kernel application
    flops_kernel2d = H_kernel * W_kernel * count_multiply_add_as

    # Calculate n_H and n_W (number of times kernel is applied in each dimension)
    n_H = (H_input + 2*H_padding - H_kernel) // H_stride + 1
    n_W = (W_input + 2*W_padding - W_kernel) // W_stride + 1

    # Calculate total number of kernel applications
    num_kernel_travel = n_H * n_W

    # Calculate flops for all channels
    total_flops = C_input * C_output * num_kernel_travel * flops_kernel2d

    return total_flops


def estimate_linear_flops(in_features, out_features, count_multiply_add_as=2):
    return count_multiply_add_as * in_features * out_features


def estimate_flops_per_token(model, dummy_shape, patch_size, count_multiply_add_as = 2, includes_backward = True):
    """
    Args:
    model (nn.Module): The convolutional neural network model.
    dummy_shape (List):
        - B (int): Batch size.
        - C (int): Number of channels in the input image.
        - H (int): Height of the input image.
        - W (int): Width of the input image.
    patch_size (int): Size of the patch to calculate FLOPs per token.

    Forward/Backward
    - num_flops_bwd    = 2 * num_flops_fwd
    - num_flops_fwdbwd = num_flops_fwd + num_flops_bwd
                       = 3 * num_flops_fwd

    """
    hooks = []
    layer_flops = []

    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            H_in, W_in = input[0].size(2), input[0].size(3)
            C_in = module.in_channels
            C_out = module.out_channels
            H_kernel, W_kernel = module.kernel_size
            H_stride, W_stride = module.stride
            H_padding, W_padding = module.padding

            flops = estimate_conv_flops(H_in, W_in, C_in, C_out, H_kernel, W_kernel, H_stride, W_stride, H_padding, W_padding, count_multiply_add_as)
            layer_flops.append(flops)
        elif isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features

            flops = estimate_linear_flops(in_features, out_features, count_multiply_add_as)
            layer_flops.append(flops)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn))

    # Use dummy data to capture all parameters for estimting a conv mfu
    B, C, H, W = dummy_shape
    dummy_input = torch.randn(B, C, H, W, device = device)

    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    for hook in hooks:
        hook.remove()

    model.train()

    total_flops = sum(layer_flops)

    # Include backward pass FLOPs if specified
    if includes_backward:
        total_flops *= 3  # Fwd pass + 2x bwd pass

    # Flops per pixel
    num_pixels = torch.tensor(dummy_shape).prod().item()
    flops_per_pixel = total_flops / num_pixels

    # Flops per token
    token_size = patch_size**2
    flops_per_token = flops_per_pixel * token_size

    return flops_per_token

dummy_shape = 1, 1, 1024, 1024  # Hard coded only to measure flops
patch_size  = model.backbone.config.patch_size
num_flops_per_token = estimate_flops_per_token(model, dummy_shape, patch_size, count_multiply_add_as = 2, includes_backward = True)


def estimate_mfu_per_iteration(num_flops_per_token, total_num_tokens_per_iteration, t_detla, peak_flops_per_sec):
    """
    Estimate model flops utilization (MFU) in units of peak FLOPS of the GPUs.
    """
    # Flops per iteration
    num_flops_per_iteration = num_flops_per_token * total_num_tokens_per_iteration

    # MFU per iteration
    num_flops_per_iteration_per_sec = num_flops_per_iteration / t_detla
    mfu = num_flops_per_iteration_per_sec / peak_flops_per_sec

    return mfu


# ----------------------------------------------------------------------- #
#  TRAINING LOOP
# ----------------------------------------------------------------------- #
batch_input_shape = None
preempt_metadata_path = os.environ.get('PREEMPT_METADATA_PATH', None)
logger.debug(f'[RANK {dist_rank}] Ready for training loop...')
iteration_counter = 0  # One iteration is one param update after one or a few forward/backward pass
try:
    # -- Loop over epochs
    # Only increment starting epoch if current epoch was fully completed
    for epoch in tqdm.tqdm(range(max_epochs), desc = f'[RANK {dist_rank}] Epoch'):
        # Skip epochs up to, but not including the last_epoch
        if epoch < last_epoch: continue

        # Reset dataset in a new epoch???
        if not from_resume:
            dataset_train.reset()

        # Otherwise, update the dataset index according to the training state
        else:
            # Update the dataset status
            dataset_train.start_idx = iter_state.get("start_idx")
            dataset_train.end_idx   = iter_state.get("end_idx")

        # -- Loop over dataset segments
        for seg in tqdm.tqdm(range(dataset_train.num_seg), desc = f'[RANK {dist_rank}] Segment'):
            # Skip previous segments up to and including the last_seg
            if epoch == last_epoch and seg <= last_seg:
                continue

            # Switch to training state
            model.train()

            # Prepare training on one segment (iteration)
            # Set next segment or break the loop when having no next segment
            requires_reset = dataset_train.set_start_idx(dataset_train.end_idx)
            if requires_reset:
                break

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.start()

            if dist_rank == 0:
                logger.info(f"Working on segment: {dataset_train.start_idx}:{dataset_train.end_idx}")

            # Split sampler across ranks
            sampler = torch.utils.data.DistributedSampler(
                dataset_train,
                shuffle   = True,
                seed      = base_seed,
                drop_last = drop_last_in_sampler
            ) if uses_dist else None
            dataloader = torch.utils.data.DataLoader(
                dataset_train,
                batch_size      = batch_size,
                sampler         = sampler,
                num_workers     = num_workers,
                collate_fn      = custom_collate,
                drop_last       = drop_last_in_loader,
                pin_memory      = pin_memory,
                prefetch_factor = prefetch_factor,
            )

            # Shuffle the training example
            if uses_dist:
                sampler.set_epoch(epoch)

            ## # [WORKAROUND]
            ## # FIXME: Better data cleaning will eliminate None batch
            ## if batch_input_shape is None:
            ##     dist.barrier()
            ##     object_list = [None, ]
            ##     if dist_rank == 0:
            ##         dataset_eval_train.reset()
            ##         dataset_eval_train.set_start_idx(0)
            ##         dataloader_eval = torch.utils.data.DataLoader(
            ##             dataset_eval_train,
            ##             batch_size  = batch_size,
            ##             sampler     = None,
            ##             num_workers = num_workers,
            ##             shuffle     = False,
            ##             collate_fn  = custom_collate,
            ##             pin_memory      = pin_memory,
            ##             prefetch_factor = prefetch_factor,
            ##         )
            ##         dataloader_eval_iter = iter(dataloader_eval)
            ##         logger.debug(f"[RANK {dist_rank}] Identifying the shape of batch_input...")
            ##         while batch_input_shape is None:
            ##             try:
            ##                 batch_data = next(dataloader_eval_iter)
            ##                 if batch_data is not None:
            ##                     batch_input, batch_target = batch_data
            ##                     batch_input_shape = batch_input.shape
            ##                     logger.debug(f"[RANK {dist_rank}] Before broadcast, Shape of batch_input = {batch_input_shape}")
            ##             except StopIteration:
            ##                 raise ValueError(f"[RANK {dist_rank}] No valid eval data found for obtaining the input shape!!!")
            ##                 break
            ##         object_list = [batch_input_shape, ]
            ##     if uses_dist:
            ##         dist.broadcast_object_list(object_list, src = 0)
            ##         batch_input_shape = object_list[0]

            # -- Loop over mini batches
            # --- Set up helper variables for gradient accum and reporting
            # Set up gradient accumulation helper variables
            grad_nosync_counter         = 0
            num_batches                 = len(dataloader)
            num_remainder_batches       = num_batches % grad_accum_steps
            start_idx_remainder_batches = num_batches - num_remainder_batches  # e.g. total=102, steps=5, idx = 102 - 102%5 = 100

            # Aggregate the loss and number of processed tokens during each gradient accumulation
            total_loss       = torch.tensor(0.0, device = device)
            total_num_tokens = torch.tensor(0.0, device = device)

            # Set a timer flag
            starts_timer = True

            # --- Mini batch loop
            logger.debug(f"[RANK {dist_rank}] Start processing {len(dataloader)} batches at epoch {epoch}, seg {seg}.")
            for batch_idx, batch_data in tqdm.tqdm(
                enumerate(dataloader),
                total = num_batches,
                desc  = f'[RANK {dist_rank}] Mini batch',
            ):
                # Start timer???
                if starts_timer:
                    t_start = time.monotonic()
                    starts_timer = False

                # ---- Forward/Backward during an iteration
                # Create dummy data for a None batch
                # FIXME: Better data cleaning will eliminate None batch
                if batch_data is None:
                    logger.debug(f"[RANK {dist_rank}] Found None batch at batch idx {batch_idx}.  Creating a dummy input!!!")
                    batch_input  = torch.zeros(batch_input_shape, dtype = mixed_precision_dtype)
                    batch_target = torch.zeros(batch_input_shape, dtype = mixed_precision_dtype)
                    batch_data = (batch_input, batch_target)

                # Concat data to perform the identical transform on input and target
                batch_data = torch.cat(batch_data, dim = 0)    # (2*B, C, H, W)
                batch_data = batch_data.to(device, non_blocking = True, dtype = mixed_precision_dtype)

                # Optional transform
                if transforms is not None:
                    for enum_idx, trans in enumerate(transforms):
                        batch_data = trans(batch_data)

                # Unpack vars
                current_batch_size = batch_data.size(0) // 2
                batch_input  = batch_data[                  :current_batch_size]
                batch_target = batch_data[current_batch_size:                  ]

                # Optionally binarize the label
                if transforms is not None:
                    batch_target = batch_target > 0.5

                # Specify the effective grad accum steps
                real_grad_accum_steps = grad_accum_steps if batch_idx < start_idx_remainder_batches else num_remainder_batches

                # Conditionally turn off grad sync for grad accumulation to simulate a larger batch unless the sync is due or the last batch
                # Refer to https://github.com/pytorch/pytorch/blob/6c4f43f82675b5fcfe8cf3e5983d0c0f326408aa/test/distributed/fsdp/test_fsdp_grad_acc.py#L180
                is_grad_sync_required = is_last_batch(batch_idx, len(dataloader)) or is_action_due(grad_nosync_counter, grad_accum_steps)
                with grad_sync_context(is_grad_sync_required):
                    # Forward
                    with autocast_context:
                        batch_output = model(batch_input)
                        loss = criterion(batch_output, batch_target)
                        loss = loss.mean()
                        loss = loss / real_grad_accum_steps  # scale the loss to account for gradient accumulation
                        logger.debug(f"[Rank {dist_rank}] loss = {loss}")

                    # Accumulate loss
                    total_loss += loss.detach()

                    # Accumulate number of tokens processed
                    total_numel = batch_input.numel()  # Get number of numeric elements
                    token_size  = model.config.backbone.patch_size**2
                    num_tokens  = total_numel / token_size
                    total_num_tokens += num_tokens

                    # Backward
                    scaler.scale(loss).backward()

                # Increment the grad nosync counter
                grad_nosync_counter += 1

                # Conditional parameter updates when grad sync is required
                if is_grad_sync_required:
                    # ---- Update neural network parameters
                    # Grad clipping
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip) \
                                    if (not uses_dist) or sharding_strategy == ShardingStrategy.NO_SHARD \
                                    else \
                                    model.clip_grad_norm_(grad_clip)

                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()

                    # ---- Report the current iteration
                    # Increment the iteration counter after param update
                    iteration_counter += 1

                    # Obtain the mean total loss
                    if uses_dist:
                        dist.all_reduce(total_loss, op = dist.ReduceOp.AVG)  # Avg across ranks

                    # Obtain the total number of tokens processed
                    if uses_dist:
                        dist.all_reduce(total_num_tokens, op = dist.ReduceOp.SUM)  # Sum across ranks

                    # Wait for all gpus to complete work
                    if device_type == "cuda":
                        torch.cuda.synchronize()

                    # Stop timer
                    t_end = time.monotonic()

                    # Calculate tokens per second
                    t_delta = t_end - t_start
                    tokens_per_sec = total_num_tokens / t_delta

                    # Log the training loop loss after a forward/backward/update
                    if dist_rank == 0:
                        # MFU
                        mfu_per_iteration = estimate_mfu_per_iteration(num_flops_per_token, total_num_tokens, t_delta, peak_flops_per_sec)

                        # Misc
                        current_lrs   = scheduler.get_lr()
                        seg_start_idx = dataset_train.start_idx
                        seg_end_idx   = dataset_train.end_idx

                        # Log
                        log_data = {
                            "rank"               : dist_rank,
                            "logevent"           : "LOSS:TRAIN",
                            "iteration"          : iteration_counter,
                            "segment"            : f"{seg_start_idx}-{seg_end_idx}",
                            "learning_rate"      : ",".join(f"{lr}" for lr in current_lrs),
                            "grad_norm"          : f"{grad_norm:.6f}",
                            "mean_train_loss"    : f"{total_loss:.6f}",
                            "tokens_per_sec"     : f"{tokens_per_sec:.1e}",
                            "mfu_per_iteration"  : f"{mfu_per_iteration:.3f}",
                            "grad_nosync_counter": grad_nosync_counter,
                        }
                        log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                        logger.info(log_msg)

                    # ---- Monitor training dynamics
                    # Do it before zero-ing gradients
                    if monitors_dynamics:
                        # Monitor preactivation and activation of the nonlinearity
                        for name, act in act_monitor.activations.items():
                            mean_preact, std_preact = act.get('pre')
                            mean_act, std_act       = act.get('pos')
                            log_data = {
                                "rank"        : dist_rank,
                                "iteration"   : iteration_counter,
                                "logevent"    : "DYNAMICS:ACT",
                                "name"        : name,
                                "preact.mean" : mean_preact,
                                "preact.std"  : std_preact,
                                "act.mean"    : mean_act,
                                "act.std"     : std_act,
                            }
                            log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                            logger.info(log_msg)

                        # Monitor param update
                        current_lr = scheduler.get_lr()[0]  # It's a list
                        backbone_param_monitor = monitor_param_update_metrics(model.backbone, current_lr)  # Motifs like transformer blocks

                        for k, v in backbone_param_monitor.get('percent_param_update').items():
                            log_data = {
                                "rank"      : dist_rank,
                                "iteration" : iteration_counter,
                                "logevent"  : "DYNAMICS:PARAMS",
                                "type"      : "encoder",
                                "name"      : k,
                                "update"    : v,
                            }
                            log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                            logger.info(log_msg)

                    # ---- Reset for the next iteration
                    # Flush the gradients
                    optimizer.zero_grad(set_to_none = True)

                    # Reset grad accum counter
                    grad_nosync_counter = 0

                    # Reset the loss accumulator
                    total_loss *= 0.0

                    # Reset the token accumulator
                    total_num_tokens *= 0

                    # Reset timer flag
                    starts_timer = True

                    # ---- Update lr every few seg (X segs = one step/iteration)
                    if is_action_due(iteration_counter, scheduler_update_iterations):
                        scheduler.step()
                        if dist_rank == 0:
                            current_lrs = scheduler.get_lr()
                            current_lrs_msg = ",".join(f"{lr}" for lr in current_lrs)
                            logger.info(f"lr is updated to {current_lrs_msg}.")

                    # ---- Eval and checkpointing
                    if is_action_due(iteration_counter, chkpt_saving_iterations):
                        # !!!!!!!!!!!!!!!
                        # !! Data dump !!
                        # !!!!!!!!!!!!!!!
                        data_dump_timestamp = {
                            "uses_dist"       : uses_dist,
                            "dist_rank"       : dist_rank,
                            "dist_world_size" : dist_world_size,
                        }
                        if data_dump_on:
                            data_dump_timestamp.update({
                                "fl_log_prefix"   : fl_log_prefix,
                                "epoch"           : epoch,
                                "seg"             : seg,
                            })

                        if dist_rank == 0:
                            logger.debug(f'[RANK {dist_rank}] Start evaluation...')

                        # ---- - Eval
                        # ---- -- Train
                        # Get a random subset of the training set
                        train_loss = torch.tensor(float('nan'))
                        num_eval_retry = 0
                        while torch.isnan(train_loss) and (num_eval_retry < max_eval_retry):
                            dataset_eval_train.reset()
                            high_seg_idx = max(dataset_eval_train.total_size - seg_size * dist_world_size, 1)
                            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
                            dataset_eval_train.set_start_idx(rand_start_idx)

                            sampler_eval = torch.utils.data.DistributedSampler(
                                dataset_eval_train,
                                shuffle   = True,
                                seed      = base_seed,
                                drop_last = drop_last_in_sampler,
                            ) if uses_dist else None
                            dataloader_eval = torch.utils.data.DataLoader(
                                dataset_eval_train,
                                batch_size      = batch_size,
                                sampler         = sampler_eval,
                                num_workers     = num_workers,
                                shuffle         = False,
                                collate_fn      = custom_collate,
                                drop_last       = drop_last_in_loader,
                                pin_memory      = pin_memory,
                                prefetch_factor = prefetch_factor,
                            )

                            # Shuffle the training example
                            if uses_dist:
                                sampler_eval.set_epoch(rand_start_idx)  # Any integer is fine

                            # Get loss
                            train_loss = estimate_loss(
                                dataloader_eval,
                                model,
                                criterion,
                                autocast_context,
                                max_iter              = max_eval_iter,
                                desc                  = '(training set)',
                                device                = device,
                                dummy_input_shape     = batch_input_shape,
                                mixed_precision_dtype = mixed_precision_dtype,
                                transforms            = transforms,
                                **data_dump_timestamp,
                            )
                            num_eval_retry += 1

                        # Log the train loss
                        if dist_rank == 0:
                            seg_start_idx = dataset_eval_train.start_idx
                            seg_end_idx   = dataset_eval_train.end_idx
                            logger.info(f"[RANK {dist_rank}] LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean train loss = {train_loss:.8f}")

                        # ---- -- Validation
                        # Get a random subset of the validation set
                        validate_loss = torch.tensor(float('nan'))
                        num_eval_retry = 0
                        while torch.isnan(validate_loss) and (num_eval_retry < max_eval_retry):
                            dataset_eval_val.reset()
                            high_seg_idx = max(dataset_eval_val.total_size - seg_size * dist_world_size, 1)
                            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
                            dataset_eval_val.set_start_idx(rand_start_idx)

                            sampler_eval = torch.utils.data.DistributedSampler(
                                dataset_eval_val,
                                shuffle   = True,
                                seed      = base_seed,
                                drop_last = drop_last_in_sampler,
                            ) if uses_dist else None
                            dataloader_eval = torch.utils.data.DataLoader(
                                dataset_eval_val,
                                batch_size      = batch_size,
                                sampler         = sampler_eval,
                                num_workers     = num_workers,
                                shuffle         = False,
                                collate_fn      = custom_collate,
                                drop_last       = drop_last_in_loader,
                                pin_memory      = pin_memory,
                                prefetch_factor = prefetch_factor,
                            )

                            # Shuffle the validation example
                            if uses_dist:
                                sampler_eval.set_epoch(rand_start_idx)  # Any integer is fine

                            validate_loss = estimate_loss(
                                dataloader_eval,
                                model,
                                criterion,
                                autocast_context,
                                max_iter              = max_eval_iter,
                                desc                  = '(validation set)',
                                device                = device,
                                dummy_input_shape     = batch_input_shape,
                                mixed_precision_dtype = mixed_precision_dtype,
                                transforms            = transforms,
                                **data_dump_timestamp,
                            )
                            num_eval_retry += 1

                        # Log the validation loss
                        if dist_rank == 0:
                            seg_start_idx = dataset_eval_val.start_idx
                            seg_end_idx   = dataset_eval_val.end_idx
                            logger.info(f"[RANK {dist_rank}] LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean validation loss = {validate_loss:.8f}")

                        # ---- - Save checkpoint
                        if validate_loss < loss_min:
                            loss_min = validate_loss

                            # Collect training state
                            iter_state["epoch"]     = epoch
                            iter_state["seg"]       = seg
                            iter_state["start_idx"] = dataset_train.start_idx
                            iter_state["end_idx"]   = dataset_train.end_idx
                            iter_state["loss_min"]  = loss_min

                            dir_chkpt = f"{timestamp}.epoch_{epoch}.end_idx_{dataset_train.end_idx}"
                            if fl_chkpt_prefix is not None: dir_chkpt = f"{fl_chkpt_prefix}.{dir_chkpt}"
                            path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
                            checkpointer.save(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt)
                            logger.info(f"Saving checkpoint at {path_chkpt}.")

                        # All ranks wait until the end of evaluation by rank 0
                        # [WARNING] Expecting NCCL TIMEOUT ERROR if the evaluation takes too long
                        if uses_dist:
                            dist.barrier()
                        logger.debug(f'[RANK {dist_rank}] Done evaluation...')

                    # ---- Preemptive checkpointing
                    if preempt_metadata_path is not None and is_action_due(iteration_counter, preempt_chkpt_saving_iterations):
                        # Collect training state
                        iter_state["epoch"]     = epoch
                        iter_state["seg"]       = seg
                        iter_state["start_idx"] = dataset_train.start_idx
                        iter_state["end_idx"]   = dataset_train.end_idx
                        iter_state["loss_min"]  = loss_min

                        dir_chkpt = f"{timestamp}.preempt"
                        if fl_chkpt_prefix is not None: dir_chkpt = f"{fl_chkpt_prefix}.{dir_chkpt}"
                        path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
                        checkpointer.save(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt)
                        logger.info(f"[RANK {dist_rank}] Saving preemptive checkpoint (epoch {epoch}, end_idx {dataset_train.end_idx}) at {path_chkpt}.")

                        if dist_rank == 0:
                            with open(preempt_metadata_path, "w") as f:
                                f.write(path_chkpt)
                            logger.info(f"[RANK {dist_rank}] Saving preemptive metadata (epoch {epoch}, end_idx {dataset_train.end_idx}) at {preempt_metadata_path}.")

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.update()

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.stop()

        # Reset last_seg
        last_seg = -1

        # Reset the from_resume flag
        from_resume = False

except KeyboardInterrupt:
    logger.error(f"[RANK {dist_rank}] Training was interrupted!")
except Exception as e:
    tb = traceback.format_exc()
    logger.error(f"[RANK {dist_rank}] Error occurred: {e}\nTraceback: {tb}")
finally:
    # Clean up hooks
    if monitors_dynamics:
        act_monitor.remove_hooks()

    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
