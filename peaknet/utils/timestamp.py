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
