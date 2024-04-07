import io
import csv
import json
import socket
import numpy as np

import torch
import torch.distributed as dist

from multiprocessing import shared_memory

from torch.utils.data import Dataset

import warnings

from dataclasses import dataclass
from typing import Optional, List, Tuple

from ..utils_fsdp import broadcast_dict
from ..perf import Timer

# ----------------------------------------------------------------------- #
#  DATALOADER FOR TRAINING BY ALL RANKS
# ----------------------------------------------------------------------- #
@dataclass
class IPCDistributedSegmentedDatasetConfig:
    """Configuration for the Remote Distributed Segmented Dataset.

    Attributes:
        full_dataset (List): The complete dataset details to be segmented and distributed.
        micro_batch_size_per_rank (int): The size of each micro-batch to be processed by each rank.
        world_size (int): Total number of distributed processes (ranks) in use.
        transforms (List): A list of transformations to apply to each data item.
        is_perf (bool): Flag to enable performance timing for transformations. Default is False.
        server_address (str): URL of the server to fetch data from. Defaults to 'http://localhost:5001'.
    """
    full_dataset             : List
    micro_batch_size_per_rank: int
    world_size               : int
    transforms               : List
    is_perf                  : bool = False
    server_address           : Tuple = ('localhost', 5000)

class IPCDistributedSegmentedDataset(Dataset):
    """A dataset class designed for fetching and distributing segments of data
    in a distributed training environment.

    This class allows for efficient data loading and processing across multiple
    distributed processes.
    """
    def __init__(self, config: IPCDistributedSegmentedDatasetConfig):
        self.full_dataset              = config.full_dataset
        self.micro_batch_size_per_rank = config.micro_batch_size_per_rank
        self.world_size                = config.world_size
        self.server_address            = config.server_address
        self.transforms                = config.transforms
        self.is_perf                   = config.is_perf
        self.total_size                = len(config.full_dataset)

        self.start_idx  = 0
        self.end_idx    = self.calculate_end_idx()

    def calculate_end_idx(self):
        # Calculate and return the end index for the current dataset segment.
        return min(self.start_idx + self.micro_batch_size_per_rank * self.world_size, self.total_size)

    def set_start_idx(self, start_idx):
        self.start_idx = start_idx
        self.end_idx = self.calculate_end_idx()

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # Ensure idx is within the bounds of the current segment
        if idx >= (self.end_idx - self.start_idx):
            raise IndexError("Index out of range for the current segment")

        # Map the local index to the correct global index within the segment
        global_idx = self.start_idx + idx

        # Obtain dataset handle
        exp, run, access_mode, detector_name, event = self.full_dataset[global_idx]

        # Fetch event
        image = self.fetch_event(exp, run, access_mode, detector_name, event)    # psana image: (H, W)

        # Apply transforms
        image_tensor = None
        if image is not None and self.transforms is not None:
            image_tensor = torch.from_numpy(image[None, None])    # (B=1, C, H, W)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = None, is_on = self.is_perf):
                    image_tensor = trans(image_tensor)

        return image_tensor[0]    # Dataloader only wants data with shape of (C, H, W)

    def save_checkpoint(self, checkpoint_path, rank):
        if rank == 0:
            checkpoint = {
                'end_idx'                  : self.end_idx,
                'micro_batch_size_per_rank': self.micro_batch_size_per_rank
            }
            torch.save(checkpoint, checkpoint_path)
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint_and_broadcast(self, checkpoint_path, rank, device):
        checkpoint = None
        if rank == 0:
            checkpoint = torch.load(checkpoint_path)
        checkpoint = broadcast_dict(checkpoint, src=0, device=device)

        if checkpoint:
            self.set_start_idx(checkpoint.get('end_idx', 0))
            if 'micro_batch_size_per_rank' in checkpoint and checkpoint['micro_batch_size_per_rank'] != self.micro_batch_size_per_rank:
                warnings.warn(f"micro_batch_size_per_rank has been changed from {checkpoint['micro_batch_size_per_rank']} to {self.micro_batch_size_per_rank}. Resetting to {checkpoint['micro_batch_size_per_rank']}.")
                self.micro_batch_size_per_rank = checkpoint['micro_batch_size_per_rank']

        if dist.is_initialized():
            dist.barrier()

    def fetch_event(self, exp, run, access_mode, detector_name, event):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)

            # Send request
            request_data = json.dumps({
                'exp'          : exp,
                'run'          : run,
                'access_mode'  : access_mode,
                'detector_name': detector_name,
                'event'        : event,
                'mode'         : 'image',
            })
            sock.sendall(request_data.encode('utf-8'))

            # Receive and process response
            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)

            # Use the JSON data to access the shared memory
            shm_name = response_json['name']
            shape    = response_json['shape']
            dtype    = np.dtype(response_json['dtype'])

            # Initialize shared memory outside of try block to ensure it's in scope for finally block
            shm = None
            try:
                # Access the shared memory
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                # Convert to numpy array (this creates a copy of the data)
                result = np.array(data_array)
            finally:
                # Ensure shared memory is closed even if an exception occurs
                if shm:
                    shm.close()
                    shm.unlink()

            # Send acknowledgment after successfully accessing shared memory
            sock.sendall("ACK".encode('utf-8'))

            return result


# ----------------------------------------------------------------------- #
#  DATALOADER FOR EVALUATION BY RANK0
# ----------------------------------------------------------------------- #
@dataclass
class IPCDatasetConfig:
    """Configuration for the Inter-Processor Communication based Dataset.

    Attributes:
        full_dataset (List): The complete dataset details to be segmented and distributed.
        transforms (List): A list of transformations to apply to each data item.
        is_perf (bool): Flag to enable performance timing for transformations. Default is False.
        server_address (str): URL of the server to fetch data from. Defaults to 'http://localhost:5001'.
    """
    full_dataset             : List
    transforms               : List
    is_perf                  : bool = False
    server_address           : Tuple = ('localhost', 5000)

class IPCDataset(Dataset):
    """A dataset class designed for fetching data through Inter-Processor
    Communication.

    This class allows for efficient data loading and processing across multiple
    distributed processes.
    """
    def __init__(self, config: IPCDatasetConfig):
        self.full_dataset   = config.full_dataset
        self.server_address = config.server_address
        self.transforms     = config.transforms
        self.is_perf        = config.is_perf

    def __len__(self):
        return len(config.full_dataset)

    def __getitem__(self, idx):
        # Obtain dataset handle
        exp, run, access_mode, detector_name, event = self.full_dataset[idx]

        # Fetch event
        image = self.fetch_event(exp, run, access_mode, detector_name, event)    # psana image: (H, W)

        # Apply transforms
        image_tensor = None
        if image is not None and self.transforms is not None:
            image_tensor = torch.from_numpy(image[None, None])    # (B=1, C, H, W)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = None, is_on = self.is_perf):
                    image_tensor = trans(image_tensor)

        return image_tensor[0]    # Dataloader only wants data with shape of (C, H, W)

    def fetch_event(self, exp, run, access_mode, detector_name, event):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)

            # Send request
            request_data = json.dumps({
                'exp'          : exp,
                'run'          : run,
                'access_mode'  : access_mode,
                'detector_name': detector_name,
                'event'        : event,
                'mode'         : 'image',
            })
            sock.sendall(request_data.encode('utf-8'))

            # Receive and process response
            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)

            # Use the JSON data to access the shared memory
            shm_name = response_json['name']
            shape    = response_json['shape']
            dtype    = np.dtype(response_json['dtype'])

            # Initialize shared memory outside of try block to ensure it's in scope for finally block
            shm = None
            try:
                # Access the shared memory
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                # Convert to numpy array (this creates a copy of the data)
                result = np.array(data_array)
            finally:
                # Ensure shared memory is closed even if an exception occurs
                if shm:
                    shm.close()
                    shm.unlink()

            # Send acknowledgment after successfully accessing shared memory
            sock.sendall("ACK".encode('utf-8'))

            return result
