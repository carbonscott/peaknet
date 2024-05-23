#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- OLCF specific imports
from maxie.plugins.olcf import init_dist_env_on_summit

# -- Basic imports
import os
import yaml
import tqdm
import signal
import argparse
import logging
import traceback

from functools  import partial
from contextlib import nullcontext
from datetime   import timedelta

# -- peaknet specific imports
from peaknet.datasets.safetensors_dataset  import PeakNetDataset
from peaknet.modeling.convnextv2_bifpn_net import PeakNetConfig, PeakNet, SegHeadConfig, SegLateralLayer
from peaknet.modeling.convnextv2_encoder   import ConvNextV2BackboneConfig, ConvNextV2Backbone
from peaknet.modeling.bifpn_config         import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from peaknet.modeling.bifpn                import BiFPNBlock
from peaknet.criterion                     import CategoricalFocalLoss
from peaknet.utils.logger                  import init_logger
from peaknet.utils.seed                    import set_seed
from peaknet.utils.misc                    import is_action_due
from peaknet.lr_scheduler                  import CosineLRScheduler
from peaknet.perf                          import Timer
from peaknet.tensor_transforms             import Pad, DownscaleLocalMean, RandomPatch, RandomRotate, RandomShift
from peaknet.utils_fsdp           import (
    MemoryMaximizer,
    verify_bfloat_support,
    TrainingStateDictConfig,
    FullStateDictCheckpointConfig,
    FullStateDictCheckpoint,
    broadcast_dict,
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
torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using float16

# -- Reporting specific imports
import colorama
colorama.init(autoreset=True)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  COMMAND LINE INTERFACE
# ----------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Load training configuration from a YAML file to a dictionary.")
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
chkpt_config        = config.get("checkpoint")
dir_root_chkpt      = chkpt_config.get("directory")
chkpt_prefix        = chkpt_config.get("prefix")
path_chkpt_prev     = chkpt_config.get("path_chkpt_prev")
chkpt_saving_period = chkpt_config.get("chkpt_saving_period")

# -- Dataset
dataset_config       = config.get("dataset")
path_train_csv       = dataset_config.get("path_train")
path_eval_csv        = dataset_config.get("path_eval")
batch_size           = dataset_config.get("batch_size")
num_workers          = dataset_config.get("num_workers")
subset_size          = dataset_config.get("subset_size")
## server_address       = dataset_config.get("server_address")
transforms_config    = dataset_config.get("transforms")
num_patch            = transforms_config.get("num_patch")
size_patch           = transforms_config.get("size_patch")
frac_shift_max       = transforms_config.get("frac_shift_max")
angle_max            = transforms_config.get("angle_max")
var_size_patch       = transforms_config.get("var_size_patch")
downscale_factors    = transforms_config.get("downscale_factors")
H_pad                = transforms_config.get("H_pad")
W_pad                = transforms_config.get("W_pad")

# -- Model
model_params              = config.get("model")
uses_random_weights       = model_params.get("uses_random_weights")
freezes_backbone          = model_params.get("freezes_backbone")
backbone_params           = model_params.get("backbone")
bifpn_params              = model_params.get("bifpn")
bifpn_block_params        = bifpn_params.get("block")
bifpn_block_bn_params     = bifpn_block_params.get("bn")
bifpn_block_fusion_params = bifpn_block_params.get("fusion")
seghead_params            = model_params.get("seg_head")
seghead_num_classes       = seghead_params.get("num_classes")

# -- Loss
loss_config      = config.get("loss")
focal_config     = loss_config.get("focal")
focal_alpha      = focal_config.get("alpha")
focal_gamma      = focal_config.get("gamma")
grad_accum_steps = max(int(loss_config.get("grad_accum_steps")), 1)

# -- Optimizer
optim_config = config.get("optim")
lr           = float(optim_config.get("lr"))
weight_decay = float(optim_config.get("weight_decay"))
grad_clip    = float(optim_config.get("grad_clip"))

# -- Scheduler
lr_scheduler_config   = config.get("lr_scheduler")
patience              = lr_scheduler_config.get("patience")
warmup_iterations     = lr_scheduler_config.get("warmup_iterations")
total_iterations      = lr_scheduler_config.get("total_iterations")
uses_prev_scheduler   = lr_scheduler_config.get("uses_prev")
min_lr                = float(lr_scheduler_config.get("min_lr"))
scheduler_step_period = lr_scheduler_config.get("scheduler_step_period")

# -- Distributed envs
dist_config            = config.get("dist")
dist_backend           = dist_config.get("backend")
uses_unique_world_seed = dist_config.get("uses_unique_world_seed")
dist_dtype             = dist_config.get("dtype")

# -- Logging
logging_config = config.get("logging")
drc_log       = logging_config.get("directory")
fl_log_prefix = logging_config.get("prefix")

# -- Misc
misc_config = config.get("misc")
max_epochs           = misc_config.get("max_epochs")
max_eval_iter        = misc_config.get("max_eval_iter")
num_gpus             = misc_config.get("num_gpus")
compiles_model       = misc_config.get("compiles_model")
data_dump_on         = misc_config.get("data_dump_on", False)

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
# torchrun doesn't work well on OLCF.  Refer to https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html#torchrun
# Thanks to the suggestion by @frobnitzem
torchrun_exists = int(os.environ.get("RANK", -1)) != -1
if not torchrun_exists: init_dist_env_on_summit()

# --- Initialize distributed environment
uses_dist = int(os.environ.get("RANK", -1)) != -1
if uses_dist:
    dist_rank       = int(os.environ["RANK"      ])
    dist_local_rank = int(os.environ["LOCAL_RANK"])
    dist_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = dist_backend,
                            rank        = dist_rank,
                            world_size  = dist_world_size,
                            timeout     = timedelta(seconds=1800),
                            init_method = "env://",)
    print(f"RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")
else:
    dist_rank       = 0
    dist_local_rank = 0
    dist_world_size = 1
    print(f"NO FSDP is used.  RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")

# --- Set up GPU device
gpu_idx = dist_local_rank % torch.cuda.device_count()    # dist_local_rank is node-centric, whereas torch.cuda.device_count() is resource-centeric (on LSF)
device = f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = dist_rank if uses_unique_world_seed else 0

# --- Set up performance utility
memmax = MemoryMaximizer() if dist_local_rank == 0 else None


# ----------------------------------------------------------------------- #
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP policy
# --- Mixed precision
mixed_precision = None
if verify_bfloat_support:
    dist_dtype = 'bfloat16'
    mixed_precision = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.bfloat16,
        buffer_dtype = torch.bfloat16,
    )
else:
    dist_dtype = 'float16'
    mixed_precision = MixedPrecision(
        param_dtype  = torch.float16,
        reduce_dtype = torch.float16,
        buffer_dtype = torch.float16,
    )

# --- Sharding strategy
sharding_strategy = ShardingStrategy.FULL_SHARD

# --- Wrapping strategy
# ---- Use built-in transformer wrap policy
# Why use transformer_auto_wrap_policy for a convnet?
# Refer to https://discuss.pytorch.org/t/tips-for-wrapping-conv-layers-in-fsdp/162883
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        ConvNextV2Backbone,
        BiFPNBlock,
        SegLateralLayer,
    },
)

# --- Activation checkpointing
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu  = False,
    checkpoint_impl = CheckpointImpl.NO_REENTRANT,
)

# --- Backward prefetch policy
backward_prefetch = BackwardPrefetch.BACKWARD_PRE


# ----------------------------------------------------------------------- #
#  LOGGING
# ----------------------------------------------------------------------- #
timestamp = None
if dist_rank == 0:
    # Fetch the current timestamp...
    timestamp = init_logger(fl_prefix = fl_log_prefix, drc_log = drc_log, returns_timestamp = True)

    # Convert dictionary to yaml formatted string...
    config_yaml = yaml.dump(config)

    # Log the config...
    logger.info(config_yaml)
timestamp = broadcast_dict(dict(timestamp=timestamp), src = 0, device = device).get('timestamp')


# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring dataset...')
# -- Seeding
base_seed  = 0
world_seed = base_seed + seed_offset
set_seed(world_seed)

# -- Custom collate to merge patch and batch dimension using concatenation
## custom_collate = lambda batch: torch.cat(batch, dim=0)  # batch of [N, C, H, W] -> [B * N, C, H, W]
## def custom_collate(batch):
##     batch_filtered = [x for x in batch if x is not None]
##     return torch.cat(batch_filtered, dim=0) if len(batch_filtered) else None
custom_collate = None

# -- Set up transformation
transforms = (
    Pad(H_pad, W_pad),
    DownscaleLocalMean(factors = downscale_factors),
    RandomPatch(num_patch = num_patch, H_patch = size_patch, W_patch = size_patch, var_H_patch = var_size_patch, var_W_patch = var_size_patch, returns_mask = False),
    RandomRotate(angle_max),
    RandomShift(frac_y_shift_max=frac_shift_max, frac_x_shift_max=frac_shift_max),
)

# -- Set up training set
# Init the dataset
cache_size    = 10
dataset_train = PeakNetDataset(path_csv = path_train_csv, transforms = transforms, cache_size = cache_size, perfs_runtime = False)

# Split sampler across ranks
sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True) if uses_dist else None
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers = num_workers, collate_fn=custom_collate)

# -- Set up eval set
# --- For training loss
dataset_eval_train = PeakNetDataset(path_csv = path_train_csv, transforms = transforms, cache_size = cache_size, perfs_runtime = False)

# --- For val loss
dataset_eval_val = PeakNetDataset(path_csv = path_eval_csv, transforms = transforms, cache_size = cache_size, perfs_runtime = False)

# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring model...')

# -- Config the model
# --- Backbone
backbone_config = ConvNextV2BackboneConfig(**backbone_params)

# --- BiFPN
bifpn_block_params["bn"    ] = BNConfig        (**bifpn_block_bn_params)
bifpn_block_params["fusion"] = FusionConfig    (**bifpn_block_fusion_params)
bifpn_params      ["block" ] = BiFPNBlockConfig(**bifpn_block_params)
bifpn_config                 = BiFPNConfig     (**bifpn_params)

# --- Seg head
seghead_config = SegHeadConfig(**seghead_params)

# --- PeakNet
peaknet_config = PeakNetConfig(
    backbone = backbone_config,
    bifpn    = bifpn_config,
    seg_head = seghead_config,
)
model = PeakNet(config = peaknet_config)

# !! Make all params trainable, a workaround for pytorch 2.0.1
torch_version = torch.__version__
torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
if version.parse(torch_version) <= version.parse("2.0.1"):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

if dist_rank == 0:
    print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# -- Mixed precision
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=mixed_precision_dtype)

# --- GradScaler
# If enabled=False scaler is a no-op
scaler = ShardedGradScaler(enabled=(dist_dtype == 'float16'))

# -- Compile the model
if compiles_model:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# -- CHECKPOINT (FULL STATE DICT)
print(f'[RANK {dist_rank}] Confguring model checkpoint...')
chkpt_config = FullStateDictCheckpointConfig(
    model           = model,
    optimizer       = None,
    lr_scheduler    = None,
    training_state  = None,
    rank            = dist_rank,
    device          = device,
    path_checkpoint = path_chkpt_prev,
)
checkpointer = FullStateDictCheckpoint(config = chkpt_config)
from_resume = path_chkpt_prev is not None
if from_resume:
    if isinstance(checkpointer, FullStateDictCheckpoint):
        # Model is loaded
        checkpointer.pre_fsdp_load()

# -- Wrapping the model in FSDP...
if uses_dist:
    # Convert BatchNorm to SyncBatchNorm...
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using FSDP...
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

    sharded_param_count = sum(p.numel() for p in model.module.parameters())
    print(f"RANK {dist_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()

# -- Optional grad sync off (to allow grad accumulation)
grad_sync_context = lambda enables_sync: nullcontext() if enables_sync else model.no_sync()


# -- [TODO] Apply activation checkpointing
ac_layer = None
if ac_layer is not None:
    check_fn = lambda submodule: isinstance(submodule, ac_layer)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn = non_reentrant_wrapper,
        check_fn              = check_fn
    )

if dist_rank == 0:
    print(f"Current timestamp: {timestamp}")


# ----------------------------------------------------------------------- #
#  CRITERION (LOSS)
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring criterion...')
criterion = CategoricalFocalLoss(alpha       = focal_alpha,
                                 gamma       = focal_gamma,
                                 num_classes = seghead_num_classes,)


# ----------------------------------------------------------------------- #
#  Optimizer
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring optimizer...')
param_iter = model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = CosineLRScheduler(optimizer         = optimizer,
                              warmup_iterations = warmup_iterations,
                              total_iterations  = total_iterations,
                              min_lr            = min_lr)
## scheduler = CosineLRScheduler(optimizer     = optimizer,
##                               warmup_epochs = warmup_epochs,
##                               total_epochs  = total_epochs,
##                               min_lr        = min_lr)


# ----------------------------------------------------------------------- #
#  CHECKPOINT (FULL STATE DICT)
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring model, optim, scheduler, training state checkpoint...')
# -- Set init training state dict
loss_min = float('inf')
training_state = TrainingStateDictConfig(
    epoch       = 0,
    loss_min    = loss_min,
)

# -- Optional resumption
last_epoch = -1
last_seg   = -1
if from_resume:
    if isinstance(checkpointer, FullStateDictCheckpoint):
        # Optimizer, scheduler are loaded
        checkpointer.post_fsdp_load(model, optimizer, scheduler, training_state)

        # Training state
        training_state = checkpointer.config.training_state
        last_epoch     = training_state.epoch
        loss_min       = training_state.loss_min

        logger.info(f"PREV - last_epoch {last_epoch}, loss_min = {loss_min}")


# ----------------------------------------------------------------------- #
#  HELPER
# ----------------------------------------------------------------------- #
@torch.no_grad()
def estimate_loss(dataloader, model, autocast_context, max_iter = None, desc = '', device = 'cpu', **kwargs):
    ''' Estimate loss.
        The dataloader should be wrapped with Dataloader class or
        DistributedSampler class, best with shuffle being true.  The shuffle
        takes place before batching.
    '''
    # -- Setup
    dist_rank       = kwargs.get('dist_rank')
    dist_world_size = kwargs.get('dist_world_size')

    if dist_rank == 0:
        print(f"[RANK {dist_rank}] - EVAL Entering")
    model.eval()

    # !!!!!!!!!!!!!!!
    # !! Data dump !!
    # !!!!!!!!!!!!!!!
    if dist_rank == 0 and data_dump_on:
        dir_data_dump = "data_dump"
        os.makedirs(dir_data_dump, exist_ok=True)

        fl_log_prefix = kwargs.get('fl_log_prefix')
        epoch         = kwargs.get('epoch')
        seg           = kwargs.get('seg')

    # -- Eval iterations
    # Set default number of iterations
    if max_iter is None:
        max_iter = len(dataloader)

    losses      = torch.zeros(len(dataloader), device = device)
    num_samples = torch.zeros(len(dataloader), device = device)
    proc_masks  = torch.zeros(len(dataloader), device = device)    # A mask to track the process
    for enum_idx, batch_data in tqdm.tqdm(enumerate(dataloader), total = max_iter, desc = f'[RANK {dist_rank}] Eval{desc}'):    # (B, C, H, W)
        # Sample at most max_iter batches
        if enum_idx >= max_iter: break

        if dist_rank == 0:
            print(f"[RANK {dist_rank}] EVAL - Pre fetching mini_batch {enum_idx}")

        # Skip a batch if it's a None
        if batch_data is None: continue

        batch_input, batch_target = batch_data
        batch_input  = batch_input.to(device, non_blocking = True)
        batch_target = batch_target.to(device, non_blocking = True)

        if dist_rank == 0:
            print(f"[RANK {dist_rank}] EVAl - Post fetching")

        with autocast_context:
            if dist_rank == 0:
                print(f"[RANK {dist_rank}] EVAL - Forwarding")
            batch_output = model(batch_input)

            if dist_rank == 0:
                print(f"[RANK {dist_rank}] EVAL - Loss")

            loss = criterion(batch_output, batch_target).mean()

        # !!!!!!!!!!!!!!!
        # !! Data dump !!
        # !!!!!!!!!!!!!!!
        if dist_rank == 0 and data_dump_on:
            mini_batch = enum_idx

            data_dump = {
                "batch_data"   : batch_data,
                "batch_output" : batch_output,
                "loss"         : loss,
            }
            path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.epoch{epoch}_seg{seg}_minib{mini_batch}.loop.pt')
            torch.save(data_dump, path_data_dump)


        losses[enum_idx]      = loss
        num_samples[enum_idx] = len(batch_input)
        proc_masks[enum_idx]   = 1

    # -- Handle nan
    # Obtain the nan mask
    non_nan_mask = ~torch.isnan(losses)

    # Get the actual mask of values that are from the processing loop and non nan
    masks = torch.logical_and(proc_masks>0, non_nan_mask)

    # -- Mean loss over eval iterations
    local_valid_losses = losses[masks].to(torch.float32)
    local_losses_mean  = local_valid_losses.mean()

    # -- Mean loss over ranks
    # Survey the occurence of nan across ranks
    world_nan_counter = torch.tensor(0, dtype = torch.int, device = device)
    local_nan_masks = torch.isnan(local_losses_mean)
    if local_nan_masks.any().item():
        print(f"[RANK {dist_rank}] EVAL ERROR: NaN encountered!!!")
        world_nan_counter += 1
        local_losses_mean  = 0.0    # Contribute to nothing in the reduced sum
    dist.all_reduce(world_nan_counter, op=dist.ReduceOp.SUM)

    # Scale the local loss for the final reduced sum
    local_losses_mean /= (dist_world_size - world_nan_counter + 1e-6)

    # Calculate reduced sum as the final mean loss
    world_losses_mean  = torch.zeros_like(local_losses_mean, dtype = torch.float32, device = device)
    world_losses_mean += local_losses_mean.to(torch.float32)
    dist.all_reduce(world_losses_mean, op=dist.ReduceOp.SUM)

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

# ----------------------------------------------------------------------- #
#  TRAINING LOOP
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Ready for training loop...')
try:
    for epoch in tqdm.tqdm(range(last_epoch+1, max_epochs), desc = f'[RANK {dist_rank}] Epoch'):
        # -- Train one epoch
        if uses_dist:
            # Shuffle the training example
            sampler_train.set_epoch(epoch)

        # [PERFORMANCE]
        if dist_local_rank == 0:
            memmax.start()

        # -- Switch to training state
        model.train()

        # -- Loop over dataset
        dataloader = dataloader_train
        grad_nosync_counter = 0  # For grad sync
        for batch_idx, batch_data in tqdm.tqdm(enumerate(dataloader), total = len(dataloader), desc = f'[RANK {dist_rank}] Mini batch'):    # (B, C, H, W)
            # -- Train one mini batch
            # Skip None batch
            # FIXME: Better data cleaning will eliminate None batch
            if batch_data is not None:
                batch_input, batch_target = batch_data
                batch_input  = batch_input.to(device, non_blocking = True)
                batch_target = batch_target.to(device, non_blocking = True)

                # Conditionally turn off grad sync for grad accumulation to simulate a larger batch unless the sync is due or the last batch
                # Refer to https://github.com/pytorch/pytorch/blob/6c4f43f82675b5fcfe8cf3e5983d0c0f326408aa/test/distributed/fsdp/test_fsdp_grad_acc.py#L180
                is_grad_sync_required = is_last_batch(batch_idx, len(dataloader)) or is_action_due(grad_nosync_counter, grad_accum_steps)
                with grad_sync_context(is_grad_sync_required):
                    # Forward
                    with autocast_context:
                        batch_output = model(batch_input)
                        loss = criterion(batch_output, batch_target).mean()
                        loss = loss / grad_accum_steps  # scale the loss to account for gradient accumulation

                    # Log the training loop loss
                    if dist_rank == 0:
                        logger.info(f"[RANK {dist_rank}] LOSS:TRAIN - epoch {epoch}, mini_batch {batch_idx}, mean train loss = {loss:.8f} grad_sync = {is_grad_sync_required}")

                    # Backward
                    scaler.scale(loss).backward()

                # Increment the grad nosync counter
                grad_nosync_counter += 1

                # Conditional parameter updates when grad sync is required
                if is_grad_sync_required:
                    # Grad clipping
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()

                    # Flush the gradients
                    optimizer.zero_grad(set_to_none=True)

                    # Reset grad accum counter
                    grad_nosync_counter = 0

        # -- Update lr every few seg (X segs = one step/iteration)
        total_step = batch_idx + epoch * len(dataloader)
        if is_action_due(total_step, scheduler_step_period):
            scheduler.step()

        # -- Eval and checkpointing
        if is_action_due(total_step, chkpt_saving_period):
            # !!!!!!!!!!!!!!!
            # !! Data dump !!
            # !!!!!!!!!!!!!!!
            data_dump_timestamp = {
                "dist_rank"       : dist_rank,
                "dist_world_size" : dist_world_size,
            }
            if data_dump_on:
                data_dump_timestamp.update({
                    "fl_log_prefix"   : fl_log_prefix,
                    "epoch"           : epoch,
                })

            if dist_rank == 0:
                print(f'[RANK {dist_rank}] Start evaluation...')

            # -- Eval
            validate_loss = 0.0

            # --- Train
            # Get a random subset of the training set
            dataset_eval_train.reset_sample_idx_map()
            high_seg_idx = len(dataset_eval_train) - subset_size * dist_world_size
            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
            dataset_eval_train.sample_subset(rand_start_idx, subset_size * dist_world_size)

            sampler_eval = torch.utils.data.DistributedSampler(dataset_eval_train, shuffle=True)
            dataloader_eval = torch.utils.data.DataLoader(dataset_eval_train, batch_size=batch_size, sampler = sampler_eval, num_workers = num_workers, shuffle = False, collate_fn=custom_collate)

            # Shuffle the training example
            sampler_eval.set_epoch(rand_start_idx)  # Any integer is fine

            # Get loss
            train_loss = estimate_loss(dataloader_eval, model, autocast_context, max_iter = max_eval_iter, desc = '(training set)', device = device, **data_dump_timestamp)

            # Log the train loss
            if dist_rank == 0:
                seg_start_idx = rand_start_idx
                seg_end_idx   = rand_start_idx + subset_size * dist_world_size
                logger.info(f"[RANK {dist_rank}] LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean train loss = {train_loss:.8f}")

            # --- Validation
            # Get a random subset of the validation set
            dataset_eval_val.reset_sample_idx_map()
            high_seg_idx = len(dataset_eval_val) - subset_size * dist_world_size
            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
            dataset_eval_val.sample_subset(rand_start_idx, subset_size * dist_world_size)

            sampler_eval = torch.utils.data.DistributedSampler(dataset_eval_val, shuffle=True)
            dataloader_eval = torch.utils.data.DataLoader(dataset_eval_val, batch_size=batch_size, sampler = sampler_eval, num_workers = num_workers, shuffle = False, collate_fn=custom_collate)

            # Shuffle the validation example
            sampler_eval.set_epoch(rand_start_idx)  # Any integer is fine

            validate_loss = estimate_loss(dataloader_eval, model, autocast_context, max_iter = max_eval_iter, desc = '(validation set)', device = device, **data_dump_timestamp)

            # Log the validation loss
            if dist_rank == 0:
                seg_start_idx = rand_start_idx
                seg_end_idx   = rand_start_idx + subset_size * dist_world_size
                logger.info(f"[RANK {dist_rank}] LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean validation loss = {validate_loss:.8f}")

            # -- Save checkpoint
            if validate_loss < loss_min:
                loss_min = validate_loss

                # Collect training state
                training_state.epoch     = epoch
                training_state.loss_min  = loss_min

                dir_chkpt = f"{timestamp}.epoch_{epoch}"
                if chkpt_prefix is not None: dir_chkpt = f"{chkpt_prefix}.{dir_chkpt}"
                path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
                checkpointer.save(model, optimizer, scheduler, training_state, path_chkpt)

            # All ranks wait until the end of evaluation by rank 0
            # [WARNING] Expecting NCCL TIMEOUT ERROR if the evaluation takes too long
            dist.barrier()
            print(f'[RANK {dist_rank}] Done evaluation...')

        # [PERFORMANCE]
        if dist_local_rank == 0:
            memmax.update()

        # [PERFORMANCE]
        if dist_local_rank == 0:
            memmax.stop()

    # Reset the from_resume flag
    from_resume = False

except KeyboardInterrupt:
    print(f"FSDP RANK {dist_rank}: Training was interrupted!")
except Exception as e:
    tb = traceback.format_exc()
    print(f"FSDP RANK {dist_rank}: Error occurred: {e}\nTraceback: {tb}")
finally:
    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
