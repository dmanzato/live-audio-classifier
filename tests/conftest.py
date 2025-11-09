"""Pytest configuration and fixtures."""
import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.device import get_device


@pytest.fixture
def device():
    """Fixture for device (CUDA, MPS, or CPU - best available)."""
    return get_device()


@pytest.fixture
def sample_spectrogram():
    """Fixture for a sample spectrogram tensor."""
    return torch.randn(2, 1, 64, 250)  # [B, C, H, W]


@pytest.fixture
def sample_waveform():
    """Fixture for a sample waveform tensor."""
    duration = 4.0
    sr = 16000
    num_samples = int(duration * sr)
    return torch.randn(1, num_samples)  # [1, T]

