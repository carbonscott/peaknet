"""
Evaluation utilities for peaknet models.

Provides sophisticated evaluation functions with distributed support,
custom criterion handling, and robust error handling.
"""

import os
import torch
import tqdm
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


def is_last_batch(batch_idx, num_batches):
    """Check if this is the last batch in the dataloader."""
    return batch_idx + 1 == num_batches


@torch.no_grad()
def estimate_loss(
    dataloader,
    model,
    criterion,
    autocast_context,
    max_iter=None,
    desc='',
    device='cpu',
    dummy_input_shape=None,
    mixed_precision_dtype=torch.float32,
    transforms=None,
    **kwargs
):
    """
    Estimate loss on a dataset with sophisticated distributed support.

    The dataloader should be wrapped with Dataloader class or
    DistributedSampler class, best with shuffle being true. The shuffle
    takes place before batching.

    Args:
        dataloader: PyTorch DataLoader for evaluation data
        model: Model to evaluate
        criterion: Loss function to use
        autocast_context: Autocast context for mixed precision
        max_iter: Maximum number of iterations to evaluate (None for full dataset)
        desc: Description for progress bar
        device: Device to run evaluation on
        dummy_input_shape: Shape for dummy data when batch is None
        mixed_precision_dtype: Data type for mixed precision
        transforms: Optional transforms to apply during evaluation
        **kwargs: Additional arguments including distributed settings and data dump options

    Returns:
        torch.Tensor: Mean loss across all batches and ranks
    """
    # -- Setup
    uses_dist = kwargs.get('uses_dist', False)
    dist_rank = kwargs.get('dist_rank', 0)
    dist_world_size = kwargs.get('dist_world_size', 1)
    data_dump_on = kwargs.get('data_dump_on', False)

    if dist_rank == 0:
        logger.debug(f"[RANK {dist_rank}] - EVAL Entering")
    model.eval()

    # !!!!!!!!!!!!!!
    # !! Data dump !!
    # !!!!!!!!!!!!!!
    if dist_rank == 0 and data_dump_on:
        dir_data_dump = "data_dump"
        os.makedirs(dir_data_dump, exist_ok=True)

        fl_log_prefix = kwargs.get('fl_log_prefix')
        step = kwargs.get('step')
        eval_type = kwargs.get('eval_type')

    # -- Eval iterations
    # Set default number of iterations
    max_iter = min(len(dataloader), max_iter or len(dataloader))

    losses = torch.zeros(len(dataloader), device=device)
    num_samples = torch.zeros(len(dataloader), device=device)
    proc_masks = torch.zeros(len(dataloader), device=device)  # A mask to track the process
    none_mask = torch.zeros(len(dataloader), device=device)  # Mask for None batches

    for enum_idx, batch_data in tqdm.tqdm(enumerate(dataloader), total=max_iter, desc=f'[RANK {dist_rank}] Eval{desc}'):    # (B, C, H, W)
        # Sample at most max_iter batches
        if enum_idx >= max_iter:
            break

        if dist_rank == 0:
            logger.debug(f"[RANK {dist_rank}] EVAL - Pre fetching mini_batch {enum_idx}")

        # -- Optional batch data transforms on GPUs to improve mfu
        # Concat data to perform the identical transform on input and target
        batch_data = torch.cat(batch_data, dim=0)    # (2*B, C, H, W)
        batch_data = batch_data.to(device, non_blocking=True, dtype=mixed_precision_dtype)

        # Optional transform
        if transforms is not None:
            for trans_idx, trans in enumerate(transforms):
                batch_data = trans(batch_data)

        # Unpack vars
        current_batch_size = batch_data.size(0) // 2
        batch_input = batch_data[:current_batch_size]
        batch_target = batch_data[current_batch_size:]

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

            # !!!!!!!!!!!!!!
            # !! Data dump !!
            # !!!!!!!!!!!!!!
            if dist_rank == 0 and data_dump_on:
                mini_batch = enum_idx

                data_dump = {
                    "batch_input": batch_input,
                    "batch_target": batch_target,
                    "batch_output": batch_output,
                }
                path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.step{step}_{eval_type}_minib{mini_batch}.fwd.pt')
                torch.save(data_dump, path_data_dump)

            if dist_rank == 0:
                logger.debug(f"[RANK {dist_rank}] EVAL - Loss")
            loss = criterion(batch_output, batch_target)
            loss = loss.mean()

        # !!!!!!!!!!!!!!
        # !! Data dump !!
        # !!!!!!!!!!!!!!
        if dist_rank == 0 and data_dump_on:
            mini_batch = enum_idx

            data_dump = {
                "batch_input": batch_input,
                "batch_target": batch_target,
                "batch_output": batch_output,
                "loss": loss,
            }
            path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.step{step}_{eval_type}_minib{mini_batch}.loop.pt')
            torch.save(data_dump, path_data_dump)

        losses[enum_idx] = loss
        num_samples[enum_idx] = len(batch_input)
        proc_masks[enum_idx] = 1

    # -- Handle nan
    # Obtain the nan mask
    non_nan_mask = ~torch.isnan(losses)

    # Get the actual mask of values that are from the processing loop and non nan
    masks = torch.logical_and(proc_masks > 0, non_nan_mask)
    masks = torch.logical_and(masks, none_mask == 0)  # Keep not-None elements

    # -- Mean loss over eval iterations
    local_valid_losses = losses[masks].to(torch.float32)
    local_losses_mean = local_valid_losses.mean()  # torch.isnan(torch.tensor([]).mean()) -> True

    # -- Mean loss over ranks
    # Survey the occurrence of nan across ranks
    world_nan_counter = torch.tensor(0, dtype=torch.int, device=device)
    local_nan_masks = torch.isnan(local_losses_mean)
    if local_nan_masks.any().item():
        logger.error(f"[RANK {dist_rank}] EVAL ERROR: NaN encountered!!!")
        world_nan_counter += 1
        local_losses_mean = 0.0    # Contribute to nothing in the reduced sum
    if uses_dist:
        dist.all_reduce(world_nan_counter, op=dist.ReduceOp.SUM)

    # Scale the local loss for the final reduced sum
    local_losses_mean /= (dist_world_size - world_nan_counter + 1e-6)

    # Calculate reduced sum as the final mean loss
    world_losses_mean = torch.zeros_like(local_losses_mean, dtype=torch.float32, device=device)
    world_losses_mean += local_losses_mean.to(torch.float32)
    if uses_dist:
        dist.all_reduce(world_losses_mean, op=dist.ReduceOp.SUM)

    # !!!!!!!!!!!!!!
    # !! Data dump !!
    # !!!!!!!!!!!!!!
    if dist_rank == 0 and data_dump_on:
        data_dump = {
            "losses": losses,
            "proc_masks": proc_masks,
            "non_nan_mask": non_nan_mask,
            "masks": masks,
            "local_valid_losses": local_valid_losses,
            "local_losses_mean": local_losses_mean,
            "world_losses_mean": world_losses_mean,
        }
        path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.step{step}_{eval_type}.end.pt')
        torch.save(data_dump, path_data_dump)

    model.train()

    return world_losses_mean


@torch.no_grad()
def estimate_loss_accelerate(
    dataloader,
    model,
    criterion,
    accelerator,
    max_iter=None,
    desc='',
    transforms=None,
    data_dump_on=False,
    **kwargs
):
    """
    Estimate loss on a dataset with native Accelerate integration.

    Uses reliable torch.distributed.all_reduce for distributed averaging
    while providing a clean Accelerate-native interface.

    Args:
        dataloader: PyTorch DataLoader for evaluation data
        model: Model to evaluate (should be accelerator.prepare'd)
        criterion: Loss function to use
        accelerator: Accelerate Accelerator object
        max_iter: Maximum number of iterations to evaluate (None for full dataset)
        desc: Description for progress bar
        transforms: Optional transforms to apply during evaluation
        data_dump_on: Enable data dumping for debugging
        **kwargs: Additional arguments for data dumping (fl_log_prefix, epoch, seg)

    Returns:
        torch.Tensor: Mean loss across all batches and ranks
    """
    # Extract data dumping parameters
    fl_log_prefix = kwargs.get('fl_log_prefix')
    step = kwargs.get('step')
    eval_type = kwargs.get('eval_type')

    if accelerator.is_main_process:
        logger.debug(f"[RANK {accelerator.process_index}] - EVAL Entering")
    model.eval()

    # Data dumping setup
    if accelerator.is_main_process and data_dump_on:
        dir_data_dump = "data_dump"
        os.makedirs(dir_data_dump, exist_ok=True)

    # Set default number of iterations
    if max_iter is None:
        max_iter = len(dataloader)

    # Initialize loss tracking tensors
    losses = torch.zeros(len(dataloader), device=accelerator.device)
    num_samples = torch.zeros(len(dataloader), device=accelerator.device)
    proc_masks = torch.zeros(len(dataloader), device=accelerator.device)
    none_mask = torch.zeros(len(dataloader), device=accelerator.device)

    # Evaluation loop with progress bar
    for enum_idx, batch_data in tqdm.tqdm(
        enumerate(dataloader), 
        total=max_iter, 
        desc=f'[RANK {accelerator.process_index}] Eval{desc}'
    ):
        # Sample at most max_iter batches
        if enum_idx >= max_iter:
            break

        if accelerator.is_main_process:
            logger.debug(f"[RANK {accelerator.process_index}] EVAL - Pre fetching mini_batch {enum_idx}")

        # Skip None batches (shouldn't happen with modern datasets but kept for safety)
        if batch_data is None:
            none_mask[enum_idx] = 1
            continue

        # Data preprocessing (same as original)
        batch_data = torch.cat(batch_data, dim=0)  # (2*B, C, H, W)
        batch_data = batch_data.to(accelerator.device, non_blocking=True)

        # Apply transforms if provided
        if transforms is not None:
            for trans in transforms:
                batch_data = trans(batch_data)

        # Split input and target
        current_batch_size = batch_data.size(0) // 2
        batch_input = batch_data[:current_batch_size]
        batch_target = batch_data[current_batch_size:]

        # Binarize target if transforms were applied
        if transforms is not None:
            batch_target = batch_target > 0.5

        if accelerator.is_main_process:
            logger.debug(f"[RANK {accelerator.process_index}] EVAL - Post fetching")

        # Forward pass (Accelerate handles mixed precision automatically)
        if accelerator.is_main_process:
            logger.debug(f"[RANK {accelerator.process_index}] EVAL - Forwarding")

        batch_output = model(batch_input)

        # Data dumping
        if accelerator.is_main_process and data_dump_on:
            data_dump = {
                "batch_input": batch_input,
                "batch_target": batch_target,
                "batch_output": batch_output,
            }
            path_data_dump = os.path.join(
                dir_data_dump, 
                f'{fl_log_prefix}.step{step}_{eval_type}_minib{enum_idx}.fwd.pt'
            )
            torch.save(data_dump, path_data_dump)

        if accelerator.is_main_process:
            logger.debug(f"[RANK {accelerator.process_index}] EVAL - Loss")

        # Calculate loss
        loss = criterion(batch_output, batch_target)
        loss = loss.mean()

        # More data dumping
        if accelerator.is_main_process and data_dump_on:
            data_dump = {
                "batch_input": batch_input,
                "batch_target": batch_target,
                "batch_output": batch_output,
                "loss": loss,
            }
            path_data_dump = os.path.join(
                dir_data_dump,
                f'{fl_log_prefix}.step{step}_{eval_type}_minib{enum_idx}.loop.pt'
            )
            torch.save(data_dump, path_data_dump)

        # Store results
        losses[enum_idx] = loss
        num_samples[enum_idx] = len(batch_input)
        proc_masks[enum_idx] = 1

    # Handle NaN detection (keep original robust logic)
    non_nan_mask = ~torch.isnan(losses)
    masks = torch.logical_and(proc_masks > 0, non_nan_mask)
    masks = torch.logical_and(masks, none_mask == 0)

    # Calculate local mean loss
    local_valid_losses = losses[masks].to(torch.float32)
    local_losses_mean = local_valid_losses.mean()

    # Distributed averaging using reliable torch.distributed (not accelerate.gather)
    world_nan_counter = torch.tensor(0, dtype=torch.int, device=accelerator.device)
    local_nan_masks = torch.isnan(local_losses_mean)
    if local_nan_masks.any().item():
        logger.error(f"[RANK {accelerator.process_index}] EVAL ERROR: NaN encountered!!!")
        world_nan_counter += 1
        local_losses_mean = 0.0

    # Use torch.distributed directly (more reliable than accelerate.gather)
    if accelerator.num_processes > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(world_nan_counter, op=dist.ReduceOp.SUM)

    # Scale local loss for final reduction
    local_losses_mean /= (accelerator.num_processes - world_nan_counter + 1e-6)

    # Calculate final mean loss across ranks
    world_losses_mean = torch.zeros_like(local_losses_mean, dtype=torch.float32, device=accelerator.device)
    world_losses_mean += local_losses_mean.to(torch.float32)

    if accelerator.num_processes > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(world_losses_mean, op=dist.ReduceOp.SUM)

    # Final data dumping
    if accelerator.is_main_process and data_dump_on:
        data_dump = {
            "losses": losses,
            "proc_masks": proc_masks,
            "non_nan_mask": non_nan_mask,
            "masks": masks,
            "local_valid_losses": local_valid_losses,
            "local_losses_mean": local_losses_mean,
            "world_losses_mean": world_losses_mean,
        }
        path_data_dump = os.path.join(dir_data_dump, f'{fl_log_prefix}.step{step}_{eval_type}.end.pt')
        torch.save(data_dump, path_data_dump)

    model.train()

    return world_losses_mean
