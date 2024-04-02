import json
import atexit
import socket
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
from peaknet.datasets.utils_psana import PsanaImg
from peaknet.perf import Timer

# Initialize buffer for each process
psana_img_buffer = {}

def get_psana_img(exp, run, access_mode, detector_name):
    """
    Fetches a PsanaImg object for the given parameters, caching the object to avoid redundant initializations.
    """
    key = (exp, run)
    if key not in psana_img_buffer:
        psana_img_buffer[key] = PsanaImg(exp, run, access_mode, detector_name)
    return psana_img_buffer[key]


## def cleanup_shared_memory(shm):
##     try:
##         shm.close()
##         shm.unlink()
##     except FileNotFoundError:
##         pass

def worker_process(server_socket):
    ## # Keep track of all shared memory objects created in this process
    ## shared_memories = []

    ## def cleanup_at_exit():
    ##     for shm in shared_memories:
    ##         cleanup_shared_memory(shm)

    ## # Register the cleanup function to run when the process exits
    ## atexit.register(cleanup_at_exit)

    while True:
        # Accept a new connection
        connection, client_address = server_socket.accept()
        try:
            # Receive request data
            request_data = connection.recv(4096).decode('utf-8')
            request_data = json.loads(request_data)
            exp = request_data.get('exp')
            run = request_data.get('run')
            access_mode = request_data.get('access_mode')
            detector_name = request_data.get('detector_name')
            event = request_data.get('event')
            mode = request_data.get('mode')

            # Fetch psana image data
            psana_img = get_psana_img(exp, run, access_mode, detector_name)
            with Timer(tag=None, is_on=True) as t:
                data = psana_img.get(event, None, mode)

            # Keep numpy array in a shared memory
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shared_array[:] = data

            ## # Add the shared memory object to the list for tracking
            ## shared_memories.append(shm)

            response_data = json.dumps({
                'name': shm.name,
                'shape': data.shape,
                'dtype': str(data.dtype)
            })

            # Send response with shared memory details
            connection.sendall(response_data.encode('utf-8'))

            # Wait for the client's acknowledgment
            ack = connection.recv(1024).decode('utf-8')
            if ack == "ACK":
                print(f"Shared memory {shm.name} ready for client to unlink.")
            else:
                print("Did not receive proper acknowledgment from client.")

        finally:
            # Cleanup if necessary
            if shm:
                shm.close()
            connection.close()

def start_server(address, num_workers):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(address)
    server_socket.listen()

    # Create and start worker processes
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker_process, args=(server_socket,))
        p.start()

# Example usage
if __name__ == "__main__":
    server_address = ('localhost', 5002)  # Example address and port
    num_workers = 10
    start_server(server_address, num_workers)
