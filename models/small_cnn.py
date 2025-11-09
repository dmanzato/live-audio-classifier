"""Small CNN model for audio classification."""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    A lightweight CNN for spectrogram-based audio classification.
    
    Architecture:
        - 3 convolutional layers with ReLU and MaxPool
        - Adaptive average pooling
        - Linear classifier
    
    Input shape: [B, 1, n_mels, time_frames]
    Output shape: [B, n_classes]
    
    Args:
        n_classes: Number of output classes.
    
    Example:
        >>> model = SmallCNN(n_classes=10)
        >>> x = torch.randn(4, 1, 64, 250)  # [B, C, H, W]
        >>> logits = model(x)  # [4, 10]
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B,128,1,1]
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [B, 1, n_mels, time_frames].
        
        Returns:
            Class logits tensor of shape [B, n_classes].
        """
        x = self.features(x)              # [B,128,1,1]
        x = x.squeeze(-1).squeeze(-1)     # [B,128]
        return self.classifier(x)         # [B,n_classes]
