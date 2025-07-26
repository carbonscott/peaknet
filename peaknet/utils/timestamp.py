#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Timestamp and checkpoint naming utilities for training scripts.

This module provides utilities for:
- Extracting timestamps from checkpoint filenames
- Parsing step numbers from checkpoint names  
- Broadcasting timestamps across distributed training ranks
"""

import os
import torch
import torch.distributed as dist


def extract_timestamp_from_checkpoint(checkpoint_path):
    """Extract timestamp from checkpoint filename.

    Examples:
    - "hiera-test_2024_0115_1430_step_12345" → "2024_0115_1430"
    - "hiera-test_2024_0115_1430_best_step_10000" → "2024_0115_1430"
    - "hiera-test_2024_0115_1430_step_12345.preempt" → "2024_0115_1430"
    
    Args:
        checkpoint_path (str): Path to checkpoint file or directory
        
    Returns:
        str or None: Extracted timestamp in format "YYYY_MMDD_HHMM", or None if not found
    """
    basename = os.path.basename(checkpoint_path)
    if "_step_" in basename:
        prefix_and_timestamp = basename.split("_step_")[0]
        parts = prefix_and_timestamp.split("_")
        if len(parts) >= 3:
            # Last 3 parts form timestamp: YYYY_MMDD_HHMM
            return "_".join(parts[-3:])
    return None


def parse_step_from_checkpoint(checkpoint_path):
    """Parse step number from any checkpoint type.

    Examples:
    - "hiera-test_2024_0115_1430_step_12345" → 12345
    - "hiera-test_2024_0115_1430_best_step_10000" → 10000
    - "hiera-test_2024_0115_1430_step_12345.preempt" → 12345
    
    Args:
        checkpoint_path (str): Path to checkpoint file or directory
        
    Returns:
        int: Extracted step number, or 0 if not found
    """
    basename = os.path.basename(checkpoint_path)
    if "step_" in basename:
        return int(basename.split("step_")[-1].split(".")[0])
    return 0


def broadcast_timestamp_to_all_ranks(timestamp, dist_rank, uses_dist):
    """Broadcast timestamp from main process to all ranks for consistency.
    
    This ensures all distributed training processes use the same timestamp
    for checkpoint naming and logging consistency.
    
    Args:
        timestamp (str): Timestamp string in format "YYYY_MMDD_HHMM" (only used on rank 0)
        dist_rank (int): Current process rank
        uses_dist (bool): Whether distributed training is being used
        
    Returns:
        str: Synchronized timestamp string across all ranks
    """
    if uses_dist and dist.is_initialized():
        if dist_rank == 0:
            # Pack timestamp as integer: 202401151430
            timestamp_int = int(timestamp.replace("_", ""))
        else:
            timestamp_int = 0

        timestamp_tensor = torch.tensor([timestamp_int], dtype=torch.long)
        if torch.cuda.is_available():
            timestamp_tensor = timestamp_tensor.cuda()
        dist.broadcast(timestamp_tensor, src=0)

        # Unpack back to format: 202401151430 → "2024_0115_1430"
        timestamp_str = str(timestamp_tensor.item()).zfill(12)  # Pad to 12 digits
        return f"{timestamp_str[:4]}_{timestamp_str[4:8]}_{timestamp_str[8:12]}"

    return timestamp