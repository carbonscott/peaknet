#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import torch.distributed as dist

from datetime import datetime

logger = logging.getLogger(__name__)

def sync_timestamp(rank, device):
    """
    Synchronize timestamp string across ranks using torch distributed.
    Format: "YYYY_MMDD_HHMM_SS"
    """
    timestamp = None
    if rank == 0:
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M_%S")

    timestamp_list = [timestamp,]
    dist.broadcast_object_list(timestamp_list, src=0, device=device)
    timestamp = timestamp_list[0]
    return timestamp


def init_logger(uses_dist, dist_rank, device, fl_prefix=None, drc_log="logs", level='info', log_to='both', logger_name=None):
    """
    Initialize logger with synchronized timestamp across distributed processes.
    
    Args:
        uses_dist (bool): Whether distributed training is used
        dist_rank (int): Rank of current process
        device (str): Device for synchronization (e.g., 'cuda:0', 'cpu')
        fl_prefix (str): Prefix for log filename
        drc_log (str): Directory for log files
        level (str): Log level ('info' or 'debug')
        log_to (str): Where to log ('file', 'console', or 'both')
        logger_name (str): Name for logger (defaults to module name)
        
    Returns:
        str: Synchronized timestamp string
    """
    # Generate and synchronize timestamp
    if uses_dist:
        timestamp = sync_timestamp(dist_rank, device)
    else:
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M_%S")

    # Use a specific logger name if provided, otherwise use module name
    logger_name = logger_name or __name__
    logger = logging.getLogger(logger_name)

    # Set to DEBUG to capture all levels
    logger.setLevel(logging.DEBUG)

    # Disable propagation to prevent affecting parent loggers
    logger.propagate = True

    # Configure log level
    log_level_spec = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    log_level = log_level_spec.get(level, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s\n%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )

    # Set up file handler if needed
    if log_to in ['file', 'both']:
        # Set up the log file path
        base_log = f"{timestamp}"
        if fl_prefix is not None:
            base_log = f"{fl_prefix}.{base_log}"
        path_log = os.path.join(drc_log, base_log)
        os.makedirs(path_log, exist_ok=True)
        fl_log = f"rank{dist_rank}.log"
        path_log = os.path.join(path_log, fl_log)

        # Create file handler
        file_handler = logging.FileHandler(path_log, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set up console handler if needed
    if log_to in ['console', 'both']:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return timestamp


def init_timestamp(uses_dist, dist_rank, device):
    """
    Initialize synchronized timestamp across distributed processes.
    
    Args:
        uses_dist (bool): Whether distributed processing is being used
        dist_rank (int): Rank of current process
        device (torch.device): Device to use for timestamp synchronization
        
    Returns:
        str: Synchronized timestamp string
    """
    # Generate and synchronize timestamp
    if uses_dist:
        timestamp = sync_timestamp(dist_rank, device)
    else:
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M_%S")

    return timestamp


class MetaLog:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ MetaLog \\___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")
