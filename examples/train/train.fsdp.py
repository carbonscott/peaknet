#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- Basic imports
import os
import yaml
import tqdm
import signal
import argparse
import logging

from functools import partial
from contextlib import contextmanager, nullcontext

# -- PeakNet specific imports
from peaknet.datasets.ipc_dataset_dist     import IPCDistributedSegmentedDatasetConfig, IPCDistributedSegmentedDataset, IPCDatasetConfig, IPCDataset
from peaknet.modeling.convnextv2_bifpn_net import PeakNetConfig, PeakNet, SegHeadConfig
from peaknet.modeling.convnextv2_encoder   import ConvNextV2BackboneConfig
from peaknet.modeling.bifpn_config         import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from peaknet.criterion                     import CategoricalFocalLoss
from peaknet.utils                         import init_logger, set_seed, is_action_due
from peaknet.lr_scheduler                  import CosineLRScheduler
from peaknet.perf                          import Timer
from peaknet.tensor_transforms             import Pad, DownscaleLocalMean, RandomPatch, RandomRotate, RandomShift
from peaknet.utils_fsdp                    import (
    MemoryMaximizer,
    verify_bfloat_support,
    TrainingStateDictConfig,
    FullStateDictCheckpointConfig,
    FullStateDictCheckpoint,
    ShardedStateDictCheckpointConfig,
    ShardedStateDictCheckpoint,
)

# -- Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim

# Libraryies used for Fully Sharded Data Parallel (FSDP)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,

    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import torch.distributed as dist

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap


torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using bfloat16
                                            # Reference: https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4

# -- Debugging specific imports
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
path_pretrain_chkpt = chkpt_config.get("pretrain")
fl_chkpt_prefix     = chkpt_config.get("filename_prefix")
dir_chkpt_prefix    = chkpt_config.get("dir_chkpt_prefix")
path_chkpt_prev     = chkpt_config.get("path_chkpt_prev")
chkpt_saving_period = chkpt_config.get("chkpt_saving_period")
epoch_unstable_end  = chkpt_config.get("epoch_unstable_end")

# -- Dataset
dataset_config    = config.get("dataset")
path_train_csv    = dataset_config.get("path_train")
path_validate_csv = dataset_config.get("path_validate")
size_batch        = dataset_config.get("batch_size")
num_workers       = dataset_config.get("num_workers")
transforms_config = dataset_config.get("transforms")
num_patch         = transforms_config.get("num_patch")
size_patch        = transforms_config.get("size_patch")
frac_shift_max    = transforms_config.get("frac_shift_max")
angle_max         = transforms_config.get("angle_max")
var_size_patch    = transforms_config.get("var_size_patch")
downscale_factors = transforms_config.get("downscale_factors")
H_pad             = transforms_config.get("H_pad")
W_pad             = transforms_config.get("W_pad")

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
loss_config  = config.get("loss")
focal_config = loss_config.get("focal")
focal_alpha  = focal_config.get("alpha")
focal_gamma  = focal_config.get("gamma")

# -- Optimizer
optim_config = config.get("optim")
lr           = float(optim_config.get("lr"))
weight_decay = float(optim_config.get("weight_decay"))
grad_clip    = float(optim_config.get("grad_clip"))

# -- Scheduler
lr_scheduler_config = config.get("lr_scheduler")
patience            = lr_scheduler_config.get("patience")
warmup_epochs       = lr_scheduler_config.get("warmup_epochs")
total_epochs        = lr_scheduler_config.get("total_epochs")
uses_prev_scheduler = lr_scheduler_config.get("uses_prev")
min_lr              = float(lr_scheduler_config.get("min_lr"))


# -- FSDP
fsdp_config            = config.get("fsdp")
fsdp_backend           = fsdp_config.get("backend")
uses_unique_world_seed = fsdp_config.get("uses_unique_world_seed")

# -- Logging
logging_config = config.get("logging")
drc_log       = logging_config.get("directory")
fl_log_prefix = logging_config.get("filename_prefix")

# -- Misc
misc_config = config.get("misc")
uses_mixed_precision = misc_config.get("uses_mixed_precision")
max_epochs           = misc_config.get("max_epochs")
num_gpus             = misc_config.get("num_gpus")
compiles_model       = misc_config.get("compiles_model")
cache_clear_after_n_batch = 5

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
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP init
# --- Initialize distributed environment
uses_fsdp = int(os.environ.get("RANK", -1)) != -1
if uses_fsdp:
    fsdp_rank       = int(os.environ["RANK"      ])
    fsdp_local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = fsdp_backend,
                            rank        = fsdp_rank,
                            world_size  = fsdp_world_size,
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

# --- Set up performance utility
memmax = MemoryMaximizer if fsdp_local_rank == 0 else None

# -- FSDP policy
# --- Mixed precision
dtype           = 'float32'
mixed_precision = None
if uses_mixed_precision:
    if verify_bfloat_support:
        dtype = 'bfloat16'
        mixed_precision = MixedPrecision(
            param_dtype  = torch.bfloat16,
            reduce_dtype = torch.bfloat16,
            buffer_dtype = torch.bfloat16,
        )
    else:
        dtype = 'float16'
        mixed_precision = MixedPrecision(
            param_dtype  = torch.float16,
            reduce_dtype = torch.float16,
            buffer_dtype = torch.float16,
        )

# --- Sharding strategy
sharding_strategy = ShardingStrategy.FULL_SHARD

# --- Wrapping strategy
min_num_params   = 500_000
auto_wrap_policy = partial(size_based_auto_wrap_policy,
                           min_num_params = min_num_params,)

# -- Logging
if fsdp_rank == 0:
    # Fetch the current timestamp...
    timestamp = init_logger(fl_prefix = fl_log_prefix, drc_log = drc_log, returns_timestamp = True)

    # Convert dictionary to yaml formatted string...
    config_yaml = yaml.dump(config)

    # Log the config...
    logger.info(config_yaml)


# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
# -- Seeding
base_seed   = 0
world_seed  = base_seed + seed_offset
set_seed(world_seed)

# -- Set up transformation
transforms = (
    Pad(H_pad, W_pad),
    DownscaleLocalMean(factors = downscale_factors),
    RandomPatch(num_patch = num_patch, H_patch = size_patch, W_patch = size_patch, var_H_patch = var_size_patch, var_W_patch = var_size_patch, returns_mask = False),
    RandomRotate(angle_max),
    RandomShift(frac_y_shift_max=frac_shift_max, frac_x_shift_max=frac_shift_max),
)

# -- Config and define the dataset
mini_batch_size_per_rank = 20
gradient_accumulation_steps = 5
micro_batch_size_per_rank = mini_batch_size_per_rank * gradient_accumulation_steps
train_dataset = [ ('xpptut15'   , 630, 'idx', 'jungfrau1M', event) for event in range(1000) ] +\
                [ ('mfxp1002121',   7, 'idx',    'Rayonix', event) for event in range(1000) ]
validate_dataset = [ ('xpptut15'   , 630, 'idx', 'jungfrau1M', event) for event in range(1000) ] +\
                   [ ('mfxp1002121',   7, 'idx',    'Rayonix', event) for event in range(1000) ]
train_config = IPCDistributedSegmentedDatasetConfig(
    full_dataset              = train_dataset,
    micro_batch_size_per_rank = micro_batch_size_per_rank,
    world_size                = fsdp_world_size,
    transforms                = transforms,
    is_perf                   = True,
    server_address            = ('localhost', 5000),
)
dataset_train = IPCDistributedSegmentedDataset(train_config)


# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
# -- Config the model
# --- Backbone
backbone_config = ConvNextV2BackboneConfig(**backbone_params)

# --- BiFPN to fuse features at different resolutions
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
model = PeakNet(config = peaknet_config)

if fsdp_rank == 0:
    print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# -- Freeze the backbone
if freezes_backbone:
    for param in model.backbone.parameters():
        param.requires_grad = False


# -- Mixed precision
dtype = 'float16'
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=mixed_precision_dtype)

# --- GradScaler. 
# If enabled=False scaler is a no-op
scaler = ShardedGradScaler(enabled=(dtype == dtype))


# -- Compile the model
if compiles_model:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# -- Wrapping the model in FSDP...
if uses_fsdp:
    # Convert BatchNorm to SyncBatchNorm...
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using FSDP...
    model = FSDP(model,
                 auto_wrap_policy  = auto_wrap_policy,
                 mixed_precision   = mixed_precision,
                 sharding_strategy = sharding_strategy,
                 device_id         = device)

    sharded_param_count = sum(p.numel() for p in model.module.parameters())
    print(f"RANK {fsdp_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()

no_sync_context = model.no_sync() if uses_fsdp else nullcontext()

if fsdp_rank == 0:
    print(f"Current timestamp: {timestamp}")


# ----------------------------------------------------------------------- #
#  CRITERION (LOSS)
# ----------------------------------------------------------------------- #
criterion = CategoricalFocalLoss(alpha       = focal_alpha,
                                 gamma       = focal_gamma,
                                 num_classes = seghead_num_classes,)


# ----------------------------------------------------------------------- #
#  Optimizer
# ----------------------------------------------------------------------- #
param_iter = model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = CosineLRScheduler(optimizer     = optimizer,
                              warmup_epochs = warmup_epochs,
                              total_epochs  = total_epochs,
                              min_lr        = min_lr)


# ----------------------------------------------------------------------- #
#  CHECKPOINT (SHARDED STATE DICT)
# ----------------------------------------------------------------------- #
# -- Training state dict
epoch_min   = 0
mini_batch  = 0
micro_batch = 0
loss_min    = float('inf')
training_state_dict_config = TrainingStateDictConfig(
    epoch       = epoch_min,
    mini_batch  = mini_batch,
    micro_batch = micro_batch,
    loss_min    = loss_min,
)

# -- Sharded state dict
chkpt_config = ShardedStateDictCheckpointConfig(
    model           = model,
    optimizer       = optimizer,
    lr_scheduler    = scheduler,
    training_state  = training_state_dict_config,
    rank            = fsdp_rank,
    device          = device,
    path_checkpoint = path_chkpt_prev,
)
checkpointer = ShardedStateDictCheckpoint(config = chkpt_config)

# -- Optional resumption
if path_chkpt_prev is not None:
    if isinstance(checkpointer, ShardedStateDictCheckpoint):
        checkpointer.load()

        training_state = checkpointer.config.training_state
        epoch_min      = training_state.epoch
        mini_batch     = training_state.mini_batch
        micro_batch    = training_state.micro_batch
        loss_min       = training_state.loss_min

        ## print(colorama.Fore.RED + f"RANK {fsdp_rank} - epoch_min   = {epoch_min  }")
        ## print(colorama.Fore.RED + f"RANK {fsdp_rank} - mini_batch  = {mini_batch }")
        ## print(colorama.Fore.RED + f"RANK {fsdp_rank} - micro_batch = {micro_batch}")
        ## print(colorama.Fore.RED + f"RANK {fsdp_rank} - loss_min    = {loss_min   }")

        mini_batch  += 1
        micro_batch += 1
        scheduler.step()    # [TODO] Need to take care of the scheduler checkpointing
        logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

## # Let's simulate saving a sharded state dict chkpt...
## training_state = TrainingStateDictConfig(**dict(
##     epoch       = 4,
##     micro_batch = 10,
##     mini_batch  = 10,
##     loss_min    = 0.2,
## ))
## if path_chkpt_prev is None:
##     path_chkpt = None
##     if fsdp_rank == 0:
##         epoch = training_state.epoch
##         micro_batch = training_state.micro_batch
##         dir_chkpt = f"{timestamp}.epoch_{epoch}.microbatch_{micro_batch}"
##         if dir_chkpt_prefix is not None: dir_chkpt = f"{dir_chkpt_prefix}.{dir_chkpt}"
##         path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
##     checkpointer.save(model, optimizer, scheduler, training_state, path_chkpt)
## 
## dist.barrier()
## if dist.is_initialized():
##     dist.destroy_process_group()

@torch.no_grad()
def estimate_loss(dataloader, model, criterion, autocast_context):
    model.eval()

    losses = torch.zeros(len(dataloader))
    for enum_idx, batch_data in enumerate(dataloader):    # (B, C, H, W)
        batch_input, batch_target = batch_data
        batch_input  = batch_input.to(device, non_blocking = True)
        batch_target = batch_target.to(device, non_blocking = True)

        with autocast_context:
            batch_output = model(batch_input)
            loss = criterion(batch_output, batch_target)
            loss = loss.mean()
            losses[enum_idx] = loss.item()

    model.train()

    return losses.mean()


# ----------------------------------------------------------------------- #
#  TRAINING LOOP
# ----------------------------------------------------------------------- #
try:
    for epoch in tqdm.tqdm(range(max_epochs)):
        # -- Restore epoch and starting micro batch index
        epoch += epoch_min

        # -- Shuffle the training examples
        if uses_fsdp:
            sampler_train.set_epoch(epoch)

        # -- Train one epoch
        while dataset_train.end_idx < len(dataset_train):
            # -- Switch to training state
            model.train()

            # -- Train one micro batch (iteration)
            # Set micro batch
            dataset_train.set_start_idx(micro_batch)    # It actually sets the end idx too, need a better name

            # Split sampler across ranks
            sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
            dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=mini_batch_size_per_rank, sampler=sampler, num_workers = num_workers)

            # Turn off grad sync for every batch to simulate a larger batch size
            with no_sync_context:
                # -- Train one mini batch
                for batch_data in dataloader:    # (B, C, H, W)
                    batch_input, batch_target = batch_data
                    batch_input  = batch_input.to(device, non_blocking = True)
                    batch_target = batch_target.to(device, non_blocking = True)

                    with autocast_context:
                        # Forward
                        batch_output = model(batch_input)
                        loss = criterion(batch_output, batch_target)
                        loss = loss.mean()
                        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

                        # Backward
                        scaler.scale(loss).backward()

                    mini_batch += 1

            # Grad clipping
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Update parameters
            scaler.step(optimizer)
            scaler.update()

            # Flush the gradients
            optimizer.zero_grad(set_to_none=True)

            # Next micro batch
            micro_batch += 1

            # -- Eval and checkpointing by rank 0
            # [TODO] Need a good strategy, maybe rank0 does quick evaluation and decide if sharded state dict should be saved
            if is_action_due(micro_batch, chkpt_saving_period) and fsdp_rank == 0:
                # -- Eval
                # --- Train
                eval_dataset = random.sample(train_dataset[:max(mini_batch, eval_sample_size)], eval_sample_size)
                eval_config = IPCDatasetConfig(
                    full_dataset   = eval_dataset,
                    transforms     = transforms,
                    is_perf        = True,
                    server_address = ('localhost', 5000),
                )
                dataset_eval = IPCDataset(eval_config)
                dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=mini_batch_size_per_rank, num_workers = num_workers, shuffle = True)
                train_loss = estimate_loss(dataloader_eval, model, criterion, autocast_context):

                # --- Validation
                eval_dataset = random.sample(validate_dataset, eval_sample_size)
                eval_config = IPCDatasetConfig(
                    full_dataset   = eval_dataset,
                    transforms     = transforms,
                    is_perf        = True,
                    server_address = ('localhost', 5000),
                )
                dataset_eval = IPCDataset(eval_config)
                dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=mini_batch_size_per_rank, num_workers = num_workers, shuffle = True)
                train_loss = estimate_loss(dataloader_eval, model, criterion, autocast_context):

            # -- Update lr after one iteration
            scheduler.step()

        # -- Reset batch
        mini_batch  = 0
        micro_batch = 0

except KeyboardInterrupt:
    print(f"FSDP RANK {fsdp_rank}: Training was interrupted!")
except Exception as e:
    print(f"FSDP RANK {fsdp_rank}: Error occurred: {e}")
finally:
    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
