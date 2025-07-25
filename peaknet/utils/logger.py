#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import torch.distributed as dist
from datetime import datetime


def _sync_timestamp():
    """
    Synchronize timestamp across distributed ranks.
    Returns local timestamp for single GPU mode.
    """
    timestamp = None
    
    if dist.is_initialized():
        # Distributed mode - synchronize timestamp
        rank = dist.get_rank()
        if rank == 0:
            timestamp = datetime.now().strftime("%Y_%m%d_%H%M_%S")
        
        timestamp_list = [timestamp]
        dist.broadcast_object_list(timestamp_list, src=0)
        timestamp = timestamp_list[0]
    else:
        # Single GPU mode - use local timestamp
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M_%S")
    
    return timestamp


def _get_rank():
    """Get current rank (0 for single GPU mode)"""
    return dist.get_rank() if dist.is_initialized() else 0


def _get_world_size():
    """Get world size (1 for single GPU mode)"""
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_distributed():
    """Check if running in distributed mode"""
    return dist.is_initialized()


def _setup_file_logging(timestamp, prefix, log_dir, rank, log_level):
    """Set up file logging for current rank"""
    # Create timestamped directory
    dir_name = timestamp
    if prefix:
        dir_name = f"{prefix}.{dir_name}"
    
    full_log_dir = os.path.join(log_dir, dir_name)
    os.makedirs(full_log_dir, exist_ok=True)
    
    # Create file handler
    log_file = os.path.join(full_log_dir, f"rank{rank}.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    
    # File formatter with detailed info
    if _is_distributed():
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [RANK %(rank)s] %(levelname)s %(name)s\n%(message)s",
            datefmt="%m/%d/%Y %H:%M:%S"
        )
        
        # Add rank to log records
        class RankFilter(logging.Filter):
            def filter(self, record):
                record.rank = rank
                return True
        
        file_handler.addFilter(RankFilter())
    else:
        # Single GPU mode - simpler format
        file_formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s\n%(message)s",
            datefmt="%m/%d/%Y %H:%M:%S"
        )
    
    file_handler.setFormatter(file_formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def _setup_console_logging(log_level, rank):
    """Set up console logging"""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Console formatter
    if _is_distributed():
        console_formatter = logging.Formatter(
            fmt="[RANK %(rank)s] %(levelname)s: %(message)s"
        )
        
        class RankFilter(logging.Filter):
            def filter(self, record):
                record.rank = rank
                return True
        
        console_handler.addFilter(RankFilter())
    else:
        # Single GPU mode - simpler format  
        console_formatter = logging.Formatter(
            fmt="%(levelname)s: %(message)s"
        )
    
    console_handler.setFormatter(console_formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)


def setup_distributed_logging(prefix=None, log_dir="logs", level="info"):
    """
    Universal logging setup that auto-detects training mode.
    
    Works seamlessly across:
    - Single GPU training
    - DDP (DistributedDataParallel) 
    - FSDP (FullyShardedDataParallel)
    
    Args:
        prefix (str): Prefix for log files and directories
        log_dir (str): Base directory for log files
        level (str): Log level ('debug', 'info', 'warning', 'error')
    
    Returns:
        str: Timestamp string for use in checkpoint naming
        
    Usage:
        import logging
        import peaknet.utils.logger as logger_utils
        
        timestamp = logger_utils.setup_distributed_logging(prefix="experiment")
        logger = logging.getLogger(__name__)  # Familiar pattern!
        logger.info("This works in all training modes")
    """
    
    # Auto-detect distributed training
    is_distributed = _is_distributed()
    rank = _get_rank()
    world_size = _get_world_size()
    
    # Generate synchronized timestamp (or local for single GPU)
    timestamp = _sync_timestamp()
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False
    
    # Set log level
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }
    log_level = level_map.get(level.lower(), logging.INFO)
    
    # Set up file logging (all ranks in distributed, rank 0 in single GPU)
    _setup_file_logging(timestamp, prefix, log_dir, rank, log_level)
    
    # Set up console logging
    if not is_distributed or rank == 0:
        # Single GPU: always log to console
        # Distributed: only rank 0 logs to console
        _setup_console_logging(log_level, rank)
    
    return timestamp


# Utility functions for distributed-aware logging
def get_rank():
    """
    Get current rank.
    
    Returns:
        int: 0 for single GPU mode, actual rank for distributed mode
    """
    return _get_rank()


def get_world_size():
    """
    Get world size.
    
    Returns:
        int: 1 for single GPU mode, actual world size for distributed mode
    """
    return _get_world_size()


def is_rank0():
    """
    Check if current process is rank 0.
    
    Returns:
        bool: True for single GPU mode, rank check for distributed mode
    """
    return _get_rank() == 0


def is_distributed():
    """
    Check if running in distributed mode.
    
    Returns:
        bool: False for single GPU, True for DDP/FSDP
    """
    return _is_distributed()


def log_on_all_ranks(logger, message, level="info"):
    """
    Smart utility to log messages on all ranks when needed.
    
    In single GPU mode: behaves like normal logger call
    In distributed mode: forces message to appear on all ranks
    
    Args:
        logger: Python logger instance (from logging.getLogger())
        message (str): Message to log
        level (str): Log level ('info', 'warning', 'error', 'debug')
        
    Usage:
        import logging
        import peaknet.utils.logger as logger_utils
        
        logger = logging.getLogger(__name__)
        logger_utils.log_on_all_ranks(logger, "Loading checkpoint...", "info")
    """
    if not _is_distributed():
        # Single GPU mode - just log normally
        getattr(logger, level)(message)
    else:
        # Distributed mode - force print on all ranks
        rank = _get_rank()
        
        # Log to file on all ranks
        getattr(logger, level)(message)
        
        # Force print to console on non-rank0 processes
        if rank != 0:
            level_name = level.upper()
            print(f"[RANK {rank}] {level_name}: {message}")


def log_rank0_only(logger, message, level="info"):
    """
    Log message only on rank 0 (or always in single GPU mode).
    
    Args:
        logger: Python logger instance
        message (str): Message to log
        level (str): Log level
    """
    if is_rank0():
        getattr(logger, level)(message)


# Backward compatibility function (deprecated but functional)
def init_logger(uses_dist=None, dist_rank=None, device=None, fl_prefix=None, drc_log="logs", level='info', **kwargs):
    """
    Backward compatibility function. Use setup_distributed_logging() instead.
    
    Note: This function ignores the manual parameters and auto-detects everything.
    """
    timestamp = setup_distributed_logging(prefix=fl_prefix, log_dir=drc_log, level=level)
    return timestamp