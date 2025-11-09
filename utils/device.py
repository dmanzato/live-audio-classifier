"""Device utility functions for selecting the best available compute device."""
import torch


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    
    Priority order:
    1. CUDA (if available)
    2. MPS (Apple Silicon, if available)
    3. CPU (fallback)
    
    Returns:
        torch.device: The best available device.
    
    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """
    Get a human-readable name for the current device.
    
    Returns:
        str: Device name (e.g., "CUDA", "MPS", "CPU").
    """
    device = get_device()
    if device.type == "cuda":
        return "CUDA"
    elif device.type == "mps":
        return "MPS (Apple Silicon)"
    else:
        return "CPU"

