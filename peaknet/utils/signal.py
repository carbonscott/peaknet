"""
Signal handling utilities for peaknet training scripts.

Provides clean signal handling for graceful shutdown of training processes.
"""

import signal


def signal_handler(signal, frame):
    """
    Signal handler that converts signals to KeyboardInterrupt.
    
    This allows the main training loop to catch the interrupt in a 
    try/except block for graceful cleanup.
    """
    # Emit Ctrl+C like signal which is then caught by our try/except block
    raise KeyboardInterrupt


def register_handlers():
    """
    Register signal handlers for graceful shutdown.
    
    Registers handlers for:
    - SIGINT (Ctrl+C)
    - SIGTERM (termination signal)
    """
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)