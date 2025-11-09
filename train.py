"""Training script for audio classification models."""
import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from models.small_cnn import SmallCNN
from datasets.urbansound8k import UrbanSound8K
from transforms.audio import SpecAugment
from utils.logging import setup_logging, get_logger
from utils.device import get_device, get_device_name

try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None

logger = get_logger("train")


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
    """
    model_name_lower = model_name.lower()
    logger.info(f"Building model: {model_name_lower} with {num_classes} classes")
    
    if model_name_lower == "smallcnn":
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
        logger.info("ResNet18 adapted for 1-channel input")
        return m
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: 'smallcnn', 'resnet18'"
        )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        loader: DataLoader for the evaluation dataset.
        device: Device to run evaluation on.
        num_classes: Number of classes.
    
    Returns:
        Tuple of (macro F1 score, confusion matrix).
    """
    model.eval()
    ys, preds = [], []
    
    try:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                ys.append(y.cpu().numpy())
                preds.append(pred.cpu().numpy())
        
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        cm = confusion_matrix(ys, preds, labels=list(range(num_classes)))
        f1 = f1_score(ys, preds, average="macro")
        return f1, cm
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array.
        out_path: Path to save the plot.
    """
    try:
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        logger.debug(f"Saved confusion matrix to {out_path}")
    except Exception as e:
        logger.error(f"Error saving confusion matrix to {out_path}: {e}", exc_info=True)
        raise


def main():
    """
    Main training function.
    
    Parses arguments, sets up datasets, trains model, and saves checkpoints.
    """
    # Setup logging
    setup_logging(level=logging.INFO)
    
    ap = argparse.ArgumentParser(description="Train audio classification model on UrbanSound8K")
    ap.add_argument("--data_root", type=str, required=True, help="Path to UrbanSound8K root")
    ap.add_argument("--train_folds", type=str, default="1,2,3,4,5,6,7,8,9", help="Comma-separated fold numbers for training")
    ap.add_argument("--val_folds", type=str, default="10", help="Comma-separated fold numbers for validation")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18"], help="Model architecture")
    ap.add_argument("--use_specaug", action="store_true", help="Use SpecAugment data augmentation")
    ap.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    ap.add_argument("--n_mels", type=int, default=64, help="Number of mel bins")
    ap.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    ap.add_argument("--hop_length", type=int, default=256, help="STFT hop length")
    ap.add_argument("--duration", type=float, default=4.0, help="Audio clip duration in seconds")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate")
    ap.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    args = ap.parse_args()
    
    # Setup logging with optional file
    if args.log_file:
        setup_logging(level=logging.INFO, log_file=args.log_file)
    
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)
    
    # Validate data root
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        sys.exit(1)
    
    # Determine device (CUDA > MPS > CPU)
    device = get_device()
    logger.info(f"Using device: {get_device_name()} ({device})")

    # Parse folds
    try:
        train_folds = [int(x.strip()) for x in args.train_folds.split(",") if x.strip()]
        val_folds = [int(x.strip()) for x in args.val_folds.split(",") if x.strip()]
    except ValueError as e:
        logger.error(f"Invalid fold specification: {e}")
        sys.exit(1)
    
    logger.info(f"Training folds: {train_folds}, Validation folds: {val_folds}")
    
    # Setup augmentation
    augment = SpecAugment() if args.use_specaug else None
    if augment:
        logger.info("Using SpecAugment data augmentation")
    
    # Load datasets
    try:
        logger.info("Loading training dataset...")
        train_ds = UrbanSound8K(
            root=str(data_root),
            folds=train_folds,
            target_sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            augment=augment,
        )
        logger.info("Loading validation dataset...")
        val_ds = UrbanSound8K(
            root=str(data_root),
            folds=val_folds,
            target_sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            augment=None,
        )
    except Exception as e:
        logger.error(f"Error loading datasets: {e}", exc_info=True)
        sys.exit(1)
    
    num_classes = len(train_ds.class_ids)
    logger.info(f"Classes: {num_classes} | Train items: {len(train_ds)} | Val items: {len(val_ds)}")

    # Create data loaders
    try:
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}", exc_info=True)
        sys.exit(1)
    
    # Build model
    try:
        model = build_model(args.model, num_classes).to(device)
        logger.info(f"Model created: {args.model} with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        logger.error(f"Error building model: {e}", exc_info=True)
        sys.exit(1)
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    logger.info(f"Optimizer: Adam, Learning rate: {args.lr}")
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    logger.info(f"Artifacts directory: {artifacts_dir.absolute()}")
    
    best_f1 = -1.0
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        # Training phase
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        
        try:
            for batch_idx, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total += x.size(0)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"  Batch {batch_idx + 1}/{len(train_dl)}, Loss: {loss.item():.4f}")
        except Exception as e:
            logger.error(f"Error during training epoch {epoch}: {e}", exc_info=True)
            continue
        
        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = total_correct / total if total > 0 else 0.0
        
        # Validation phase
        try:
            f1, cm = evaluate(model, val_dl, device, num_classes)
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} val_f1_macro={f1:.3f}"
            )
        except Exception as e:
            logger.error(f"Error during validation epoch {epoch}: {e}", exc_info=True)
            continue
        
        # Save confusion matrix
        try:
            cm_path = artifacts_dir / f"confusion_matrix_epoch{epoch}.png"
            plot_confusion_matrix(cm, str(cm_path))
            logger.debug(f"Saved confusion matrix to {cm_path}")
        except Exception as e:
            logger.warning(f"Could not save confusion matrix: {e}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            try:
                checkpoint_path = artifacts_dir / "best_model.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model (F1={f1:.3f}) to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving model checkpoint: {e}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info(f"Training completed. Best F1 score: {best_f1:.3f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
