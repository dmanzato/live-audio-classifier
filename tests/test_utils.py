"""Tests for utility functions."""
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTensorUtils:
    """Test tensor utility functions."""
    
    def test_tensor_shapes(self):
        """Test that we can create tensors with expected shapes."""
        # Test spectrogram shape
        batch_size = 2
        n_mels = 64
        time_frames = 250
        
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        assert x.shape == (batch_size, 1, n_mels, time_frames)
    
    def test_device_placement(self):
        """Test tensor device placement."""
        x = torch.randn(2, 3)
        assert x.device.type == 'cpu'
        
        if torch.cuda.is_available():
            x_cuda = x.to('cuda')
            assert x_cuda.device.type == 'cuda'
        
        if torch.backends.mps.is_available():
            x_mps = x.to('mps')
            assert x_mps.device.type == 'mps'
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        x = torch.randn(2, 3, requires_grad=True)
        y = x.sum()
        y.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

