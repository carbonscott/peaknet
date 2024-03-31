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
from contextlib import contextmanager

# -- PeakNet specific imports
from peaknet.datasets.safetensors_dataset_dist  import DistributedSegmentedDatasetConfig, DistributedSegmentedDataset
from peaknet.datasets.safetensors_dataset  import PeakNetDataset
from peaknet.modeling.convnextv2_bifpn_net import PeakNetConfig, PeakNet, SegHeadConfig
from peaknet.modeling.convnextv2_encoder   import ConvNextV2BackboneConfig
from peaknet.modeling.bifpn_config         import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from peaknet.criterion                     import CategoricalFocalLoss
from peaknet.utils                         import init_logger, split_dataset, set_seed, init_weights
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
#  MAIN
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


# Set global seed
base_seed   = 0
world_seed  = base_seed + seed_offset
set_seed(world_seed)


# ----------------------------------------------------------------------- #
#  TEST
# ----------------------------------------------------------------------- #
# Create dataset configuration
micro_batch_size_per_rank = 20
mini_batch_size_per_rank = 4
config = DistributedSegmentedDatasetConfig(
    full_dataset              = list(range(int(1e6))),
    micro_batch_size_per_rank = micro_batch_size_per_rank,
    world_size                = fsdp_world_size,
)

# Initialize the dataset
dataset = DistributedSegmentedDataset(config)

# Define checkpoint path
checkpoint_path = "junk.segmented_dataset_checkpoint.pth"
output_file_path = f"junk.global_indices_rank_{fsdp_rank}.txt"

if True:
    dataset.load_checkpoint_and_broadcast(checkpoint_path, fsdp_rank, device)
    checkpoint_path = "junk.segmented_dataset_checkpoint.02.pth"
    output_file_path = f"junk.global_indices_rank_{fsdp_rank}.02.txt"

start_idx = dataset.start_idx
with open(output_file_path, 'w') as f:
    for segment_idx in range(200):
        dataset.set_start_idx(start_idx)
        start_idx = dataset.end_idx

        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False, num_replicas=fsdp_world_size, rank=fsdp_rank)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size_per_rank, sampler=sampler)

        for batch_idx, (global_idx, val) in enumerate(dataloader):
            # Accessing the global_idx for each batch
            f.write(f"{global_idx}, {val}\n")

        if segment_idx == 101:  # Checkpoint saving condition
            print(f"Rank {fsdp_rank}: Saving checkpoint at segment {segment_idx}")
            dataset.save_checkpoint(checkpoint_path, fsdp_rank)
            dist.barrier()
            break

print(f"Rank {fsdp_rank}: Indices exported to {output_file_path}")

## # Manually iterating over segments to simulate training loop
## for segment_idx in range(int(1e4)):
##     dataset.set_segment(segment_idx)
##     sampler = torch.utils.data.DistributedSampler(dataset)
##     dataloader = torch.utils.data.DataLoader(dataset, batch_size=micro_batch_size, sampler=sampler)
## 
##     for data in dataloader:
##         # Pretend to work on mini batch
##         pass
## 
##     if segment_idx == 101:  # Checkpoint saving condition
##         print(f"[{fsdp_rank}] Prepare saving...")
##         dataset.save_checkpoint(checkpoint_path, fsdp_rank)
##         print(f"[{fsdp_rank}] Done saving...")
##         dist.barrier()
##         break


## # Load checkpoint
## dataset.load_checkpoint_and_broadcast(checkpoint_path, fsdp_rank, device)
## 
## # Verify the loaded state
## print(f"[{fsdp_rank}] Loaded segment index: {dataset.current_segment_index}")
## print(f"[{fsdp_rank}] Loaded micro batch size: {dataset.micro_batch_size}")
## print(f"[{fsdp_rank}] Loaded start idx: {dataset.start_idx}")

dist.barrier()
print(f"[{fsdp_rank}] Ending program...")
if dist.is_initialized():
    dist.destroy_process_group()
