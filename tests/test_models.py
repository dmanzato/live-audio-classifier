"""Tests for model architectures."""
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.small_cnn import SmallCNN


class TestSmallCNN:
    """Test SmallCNN model."""
    
    def test_model_creation(self):
        """Test that model can be created with different numbers of classes."""
        for n_classes in [10, 5, 20]:
            model = SmallCNN(n_classes=n_classes)
            assert model is not None
            assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        n_mels = 64
        time_frames = 250  # Approximate for 4 seconds at 16kHz
        n_classes = 10
        
        model = SmallCNN(n_classes=n_classes)
        model.eval()
        
        # Input: [B, 1, n_mels, time]
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        
        with torch.no_grad():
            output = model(x)
        
        # Output should be [B, n_classes]
        assert output.shape == (batch_size, n_classes)
    
    def test_forward_pass_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        model = SmallCNN(n_classes=10)
        model.eval()
        
        # Test with different time frame sizes
        test_cases = [
            (64, 100),   # Short
            (64, 250),   # Medium
            (64, 500),   # Long
        ]
        
        for n_mels, time_frames in test_cases:
            x = torch.randn(2, 1, n_mels, time_frames)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 10)
    
    def test_model_gradient_flow(self):
        """Test that gradients can flow through the model."""
        model = SmallCNN(n_classes=10)
        model.train()
        
        x = torch.randn(2, 1, 64, 250, requires_grad=True)
        output = model(x)
        
        # Create a dummy loss
        target = torch.randint(0, 10, (2,))
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_model_device_placement(self):
        """Test model can be moved to different devices."""
        model = SmallCNN(n_classes=10)
        
        # Test CPU
        model_cpu = model.to('cpu')
        x_cpu = torch.randn(1, 1, 64, 250)
        with torch.no_grad():
            output_cpu = model_cpu(x_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            x_cuda = torch.randn(1, 1, 64, 250).to('cuda')
            with torch.no_grad():
                output_cuda = model_cuda(x_cuda)
            assert output_cuda.device.type == 'cuda'
        
        # Test MPS if available (Apple Silicon)
        if torch.backends.mps.is_available():
            model_mps = model.to('mps')
            x_mps = torch.randn(1, 1, 64, 250).to('mps')
            with torch.no_grad():
                output_mps = model_mps(x_mps)
            assert output_mps.device.type == 'mps'

