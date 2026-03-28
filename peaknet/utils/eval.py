"""
Evaluation utilities for peaknet models.

Provides a single core evaluation function (`evaluate_model`) with distributed
support, custom criterion handling, and robust NaN handling. Thin wrappers
(`estimate_loss`, `estimate_loss_accelerate`) preserve backwards compatibility
for both the original and Accelerate-based training scripts.
"""

import os
import contextlib
import torch
import tqdm
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


def is_last_batch(batch_idx, num_batches):
    """Check if this is the last batch in the dataloader."""
    return batch_idx + 1 == num_batches


# -- Core evaluation function ------------------------------------------------

@torch.no_grad()
def evaluate_model(
    dataloader,
    model,
    criterion,
    device,
    num_processes=1,
    process_index=0,
    max_iter=None,
    mixed_precision_dtype=torch.float32,
    autocast_context_fn=None,
    transforms=None,
    data_dump_on=False,
    **kwargs
):
    """
    Evaluate a model on a dataset with distributed support and NaN handling.

    This is the single core evaluation function. Both ``estimate_loss`` (for
    manual distributed scripts) and ``estimate_loss_accelerate`` (for Accelerate
    scripts) delegate to this function.

    Args:
        dataloader: PyTorch DataLoader for evaluation data.
        model: Model to evaluate.
        criterion: Loss function.
        device: Torch device.
        num_processes: Total number of distributed ranks (1 = single-GPU).
        process_index: This rank's index.
        max_iter: Max batches to evaluate (None = full dataset).
        mixed_precision_dtype: Dtype for input data casting.
        autocast_context_fn: Callable returning a context manager for
            mixed-precision forward pass.  Called once per batch so that
            single-use (generator-based) context managers work correctly.
            ``None`` means no autocast (uses ``contextlib.nullcontext``).
        transforms: Optional list of transforms applied to concatenated data.
        data_dump_on: Whether to dump intermediate tensors for debugging.
        **kwargs: Data-dump params (``fl_log_prefix``, ``step``, ``eval_type``)
            and any extra keys (silently ignored).

    Returns:
        torch.Tensor: Mean loss across all batches and ranks.
    """
    if autocast_context_fn is None:
        autocast_context_fn = contextlib.nullcontext

    fl_log_prefix = kwargs.get('fl_log_prefix')
    step = kwargs.get('step')
    eval_type = kwargs.get('eval_type')

    if process_index == 0:
        logger.debug(f"[RANK {process_index}] - EVAL Entering")
    model.eval()

    # Data dump setup
    if process_index == 0 and data_dump_on:
        dir_data_dump = "data_dump"
        os.makedirs(dir_data_dump, exist_ok=True)

    # Clamp max_iter to dataloader length (original behavior)
    max_iter = min(len(dataloader), max_iter or len(dataloader))

    # Pre-allocate tracking tensors
    losses = torch.zeros(len(dataloader), device=device)
    num_samples = torch.zeros(len(dataloader), device=device)
    proc_masks = torch.zeros(len(dataloader), device=device)
    none_mask = torch.zeros(len(dataloader), device=device)

    for enum_idx, batch_data in tqdm.tqdm(
        enumerate(dataloader),
        total=max_iter,
        desc=f'[RANK {process_index}] Eval{kwargs.get("desc", "")}',
    ):
        if enum_idx >= max_iter:
            break

        if process_index == 0:
            logger.debug(f"[RANK {process_index}] EVAL - Pre fetching mini_batch {enum_idx}")

        # Skip None batches (defensive; shouldn't happen with modern datasets)
        if batch_data is None:
            none_mask[enum_idx] = 1
            continue

        # Concat input+target, cast to mixed_precision_dtype, move to device
        batch_data = torch.cat(batch_data, dim=0)  # (2*B, C, H, W)
        batch_data = batch_data.to(device, non_blocking=True, dtype=mixed_precision_dtype)

        # Optional transforms
        if transforms is not None:
            for trans in transforms:
                batch_data = trans(batch_data)

        # Unpack input / target
        current_batch_size = batch_data.size(0) // 2
        batch_input = batch_data[:current_batch_size]
        batch_target = batch_data[current_batch_size:]

        # Binarize target when transforms are active
        if transforms is not None:
            batch_target = batch_target > 0.5

        if process_index == 0:
            logger.debug(f"[RANK {process_index}] EVAL - Post fetching")

        # Forward pass + loss inside autocast
        with autocast_context_fn():
            if process_index == 0:
                logger.debug(f"[RANK {process_index}] EVAL - Forwarding")

            batch_output = model(batch_input)

            # Data dump (forward)
            if process_index == 0 and data_dump_on:
                data_dump = {
                    "batch_input": batch_input,
                    "batch_target": batch_target,
                    "batch_output": batch_output,
                }
                path_data_dump = os.path.join(
                    dir_data_dump,
                    f'{fl_log_prefix}.step{step}_{eval_type}_minib{enum_idx}.fwd.pt',
                )
                torch.save(data_dump, path_data_dump)

            if process_index == 0:
                logger.debug(f"[RANK {process_index}] EVAL - Loss")
            loss = criterion(batch_output, batch_target)
            loss = loss.mean()

        # Data dump (loop)
        if process_index == 0 and data_dump_on:
            data_dump = {
                "batch_input": batch_input,
                "batch_target": batch_target,
                "batch_output": batch_output,
                "loss": loss,
            }
            path_data_dump = os.path.join(
                dir_data_dump,
                f'{fl_log_prefix}.step{step}_{eval_type}_minib{enum_idx}.loop.pt',
            )
            torch.save(data_dump, path_data_dump)

        losses[enum_idx] = loss
        num_samples[enum_idx] = len(batch_input)
        proc_masks[enum_idx] = 1

    # -- NaN handling ---------------------------------------------------------
    non_nan_mask = ~torch.isnan(losses)
    masks = torch.logical_and(proc_masks > 0, non_nan_mask)
    masks = torch.logical_and(masks, none_mask == 0)

    local_valid_losses = losses[masks].to(torch.float32)
    local_losses_mean = local_valid_losses.mean()

    # -- Distributed averaging ------------------------------------------------
    uses_dist = num_processes > 1 and dist.is_initialized()

    world_nan_counter = torch.tensor(0, dtype=torch.int, device=device)
    local_nan_masks = torch.isnan(local_losses_mean)
    if local_nan_masks.any().item():
        logger.error(f"[RANK {process_index}] EVAL ERROR: NaN encountered!!!")
        world_nan_counter += 1
        local_losses_mean = 0.0
    if uses_dist:
        dist.all_reduce(world_nan_counter, op=dist.ReduceOp.SUM)

    local_losses_mean /= (num_processes - world_nan_counter + 1e-6)

    world_losses_mean = torch.zeros_like(local_losses_mean, dtype=torch.float32, device=device)
    world_losses_mean += local_losses_mean.to(torch.float32)
    if uses_dist:
        dist.all_reduce(world_losses_mean, op=dist.ReduceOp.SUM)

    # Data dump (end)
    if process_index == 0 and data_dump_on:
        data_dump = {
            "losses": losses,
            "proc_masks": proc_masks,
            "non_nan_mask": non_nan_mask,
            "masks": masks,
            "local_valid_losses": local_valid_losses,
            "local_losses_mean": local_losses_mean,
            "world_losses_mean": world_losses_mean,
        }
        path_data_dump = os.path.join(
            dir_data_dump,
            f'{fl_log_prefix}.step{step}_{eval_type}.end.pt',
        )
        torch.save(data_dump, path_data_dump)

    model.train()
    return world_losses_mean


# -- Backwards-compatible wrappers -------------------------------------------

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
    Backwards-compatible wrapper for manual-distributed training scripts.

    Extracts ``uses_dist`` / ``dist_rank`` / ``dist_world_size`` from
    ``**kwargs`` and maps them to the ``evaluate_model`` interface.
    """
    uses_dist = kwargs.pop('uses_dist', False)
    dist_rank = kwargs.pop('dist_rank', 0)
    dist_world_size = kwargs.pop('dist_world_size', 1)

    return evaluate_model(
        dataloader,
        model,
        criterion,
        device=device,
        # Only advertise multiple processes when uses_dist is True,
        # so that evaluate_model's dist guard matches the caller's intent.
        num_processes=dist_world_size if uses_dist else 1,
        process_index=dist_rank,
        max_iter=max_iter,
        mixed_precision_dtype=mixed_precision_dtype,
        # The original autocast_context (e.g. torch.amp.autocast) is reentrant,
        # so returning the same object from the factory is safe.
        autocast_context_fn=lambda: autocast_context,
        transforms=transforms,
        desc=desc,
        **kwargs,
    )


# Mapping from Accelerator.mixed_precision string to torch dtype.
_ACCEL_DTYPE_MAP = {
    'no':   torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


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
    Backwards-compatible wrapper for Accelerate-based training scripts.

    Extracts ``device``, ``num_processes``, ``process_index``,
    ``autocast_context``, and ``mixed_precision_dtype`` from the
    ``accelerator`` object and delegates to ``evaluate_model``.
    """
    mp_key = getattr(accelerator, 'mixed_precision', 'no') or 'no'
    mixed_precision_dtype = _ACCEL_DTYPE_MAP.get(mp_key, torch.float32)

    return evaluate_model(
        dataloader,
        model,
        criterion,
        device=accelerator.device,
        num_processes=accelerator.num_processes,
        process_index=accelerator.process_index,
        max_iter=max_iter,
        mixed_precision_dtype=mixed_precision_dtype,
        autocast_context_fn=accelerator.autocast,
        transforms=transforms,
        data_dump_on=data_dump_on,
        desc=desc,
        **kwargs,
    )
