#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import socket
import time
import requests
import io
import numpy as np

from multiprocessing import shared_memory

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class IPCRemotePsanaDataset(Dataset):
    def __init__(self, server_address, requests_list):
        """
        server_address: The address of the server. For UNIX sockets, this is the path to the socket.
                        For TCP sockets, this could be a tuple of (host, port).
        requests_list: A list of tuples. Each tuple should contain:
                       (exp, run, access_mode, detector_name, event)
        """
        self.server_address = server_address
        self.requests_list = requests_list

    def __len__(self):
        return len(self.requests_list)

    def __getitem__(self, idx):
        request = self.requests_list[idx]
        return self.fetch_event(*request)

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
            shape = response_json['shape']
            dtype = np.dtype(response_json['dtype'])

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

# Usage example
server_address = ('localhost', 5002)
requests_list = [ ('xpptut15'   , 630, 'idx', 'jungfrau1M', event) for event in range(1000) ] +\
                [ ('mfxp1002121',   7, 'idx',    'Rayonix', event) for event in range(1000) ]

dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)

dataloader = DataLoader(dataset, batch_size=20, num_workers=10, prefetch_factor = None)
dataloader_iter = iter(dataloader)
batch_idx       = 0
while True:
    try:
        t_s = time.monotonic()
        batch_data = next(dataloader_iter)
        t_e = time.monotonic()
        loading_time_in_sec = (t_e - t_s)

        print(f"Batch idx: {batch_idx:d}, Total time: {loading_time_in_sec:.2f} s, Average time: {loading_time_in_sec / len(batch_data) * 1e3:.2f} ms/event, Batch shape: {batch_data.shape}")

        batch_idx += 1
    except StopIteration:
        break
