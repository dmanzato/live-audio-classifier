"""Shared model building utilities for live-audio-classifier."""
import torch.nn as nn
from utils.logging import get_logger

logger = get_logger("models")

try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Build a model architecture.
    
    Args:
        model_name: Name of the model architecture ('smallcnn' or 'resnet18').
        num_classes: Number of output classes.
    
    Returns:
        Initialized model.
    
    Raises:
        RuntimeError: If resnet18 is requested but torchvision is not available.
        ValueError: If model_name is not recognized.
    
    Example:
        >>> from utils.models import build_model
        >>> model = build_model("smallcnn", num_classes=10)
        >>> x = torch.randn(4, 1, 64, 250)
        >>> logits = model(x)  # [4, 10]
    """
    model_name_lower = model_name.lower()
    logger.debug(f"Building model: {model_name_lower} with {num_classes} classes")
    
    if model_name_lower == "smallcnn":
        from models.small_cnn import SmallCNN
        return SmallCNN(n_classes=num_classes)
    elif model_name_lower == "resnet18":
        if resnet18 is None:
            raise RuntimeError(
                "torchvision not available; cannot use resnet18. "
                "Install with: pip install torchvision"
            )
        m = resnet18(weights=None)
        # Adapt first conv to 1-channel input
        if m.conv1.in_channels != 1:
            m.conv1 = nn.Conv2d(
                1, m.conv1.out_channels,
                kernel_size=m.conv1.kernel_size,
                stride=m.conv1.stride,
                padding=m.conv1.padding,
                bias=False
            )
        # Adaptive avgpool is already present; set classifier
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        logger.debug("ResNet18 adapted for 1-channel input")
        return m
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: 'smallcnn', 'resnet18'"
        )

