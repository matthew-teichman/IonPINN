import torch
import deepxde as dde
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Detects if an NVIDIA GPU is available and returns the appropriate torch device.
    Configures DeepXDE to use the same backend and device.
    """
    if os.environ.get("DDE_BACKEND") != "pytorch":
        os.environ["DDE_BACKEND"] = "pytorch"
        logger.info("Set DDE_BACKEND to pytorch.")
        
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA is available. Switching to GPU: {gpu_name}")
        # Configure PyTorch default tensor type to float32 on CUDA for DeepXDE
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available. Falling back to CPU.")
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)
        
    return device
