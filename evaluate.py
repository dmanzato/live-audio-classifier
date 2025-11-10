#!/usr/bin/env python3
"""
Evaluate and compare model performance on UrbanSound8K test set.

This script loads trained models and evaluates them on a test fold,
computing accuracy, macro F1, per-class F1, and confusion matrices.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

from datasets.urbansound8k import UrbanSound8K
from utils.device import get_device, get_device_name
from utils.logging import setup_logging, get_logger
from utils.models import build_model

logger = get_logger("evaluate")


def load_model(checkpoint_path: Path, model_name: str, num_classes: int, device: torch.device):
    """Load a trained model from checkpoint."""
    model = build_model(model_name, num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list):
    """Evaluate a model and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create per-class metrics DataFrame
    per_class_metrics = pd.DataFrame({
        'Class': class_names,
        'F1': per_class_f1,
    })
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'per_class_metrics': per_class_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare model performance on UrbanSound8K test set"
    )
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to UrbanSound8K root directory")
    parser.add_argument("--test_folds", type=str, default="10",
                       help="Comma-separated fold numbers for testing [default: 10]")
    parser.add_argument("--checkpoint_smallcnn", type=str,
                       default="artifacts/best_model_smallcnn.pt",
                       help="Path to SmallCNN checkpoint. If not found, tries artifacts/best_model.pt")
    parser.add_argument("--checkpoint_resnet18", type=str,
                       default="artifacts/best_model_resnet18.pt",
                       help="Path to ResNet18 checkpoint. If not found, tries artifacts/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation [default: 32]")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate [default: 16000]")
    parser.add_argument("--n_mels", type=int, default=64, help="Number of mel bins [default: 64]")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size [default: 1024]")
    parser.add_argument("--hop_length", type=int, default=256, help="STFT hop length [default: 256]")
    parser.add_argument("--duration", type=float, default=4.0, help="Audio duration [default: 4.0]")
    parser.add_argument("--models", type=str, default="smallcnn,resnet18",
                       help="Comma-separated list of models to evaluate [default: smallcnn,resnet18]")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file for detailed results [default: None]")
    
    args = parser.parse_args()
    
    setup_logging()
    
    device = get_device()
    logger.info(f"Using device: {get_device_name()} ({device})")
    
    # Parse test folds
    test_folds = [int(f.strip()) for f in args.test_folds.split(",")]
    logger.info(f"Test folds: {test_folds}")
    
    # Load test dataset
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        sys.exit(1)
    
    logger.info("Loading test dataset...")
    test_ds = UrbanSound8K(
        root=str(data_root),
        folds=test_folds,
        target_sr=args.sr,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        augment=None,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    num_classes = len(test_ds.class_ids)
    class_names = [test_ds.idx2name[i] for i in range(num_classes)]
    
    logger.info(f"Test dataset: {len(test_ds)} samples, {num_classes} classes")
    logger.info(f"Class names: {class_names}")
    
    # Parse models to evaluate
    models_to_eval = [m.strip().lower() for m in args.models.split(",")]
    
    results = {}
    model_objects = {}  # Store model objects for parameter counting
    
    # Evaluate each model
    for model_name in models_to_eval:
        if model_name not in ["smallcnn", "resnet18"]:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        
        checkpoint_path = Path(args.checkpoint_smallcnn if model_name == "smallcnn" else args.checkpoint_resnet18)
        
        # Fallback to default checkpoint name if specified path doesn't exist
        if not checkpoint_path.exists():
            default_checkpoint = Path("artifacts/best_model.pt")
            if default_checkpoint.exists():
                logger.info(f"Using fallback checkpoint: {default_checkpoint}")
                checkpoint_path = default_checkpoint
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path} or {default_checkpoint}, skipping {model_name}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            model = load_model(checkpoint_path, model_name, num_classes, device)
            num_params = sum(p.numel() for p in model.parameters())
            model_objects[model_name] = model
            logger.info(f"Model parameters: {num_params:,}")
            
            logger.info("Running evaluation...")
            metrics = evaluate_model(model, test_loader, device, class_names)
            results[model_name] = metrics
            
            logger.info(f"\n{model_name.upper()} Results:")
            logger.info(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            logger.info(f"  Macro F1:     {metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
            
            logger.info(f"\nPer-class F1 scores:")
            for _, row in metrics['per_class_metrics'].iterrows():
                logger.info(f"  {row['Class']:20s}: {row['F1']:.4f}")
                
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}", exc_info=True)
            continue
    
    # Print comparison
    if len(results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"{'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Params':<15}")
        logger.info("-" * 60)
        
        for model_name in models_to_eval:
            if model_name in results:
                m = results[model_name]
                num_params = sum(p.numel() for p in model_objects[model_name].parameters())
                logger.info(
                    f"{model_name.upper():<15} "
                    f"{m['accuracy']*100:>6.2f}%     "
                    f"{m['macro_f1']:>6.4f}     "
                    f"{m['weighted_f1']:>6.4f}     "
                    f"{num_params:>13,}"
                )
        
        # Determine winner
        if len(results) == 2:
            models = list(results.keys())
            if results[models[0]]['macro_f1'] > results[models[1]]['macro_f1']:
                winner = models[0]
            else:
                winner = models[1]
            logger.info(f"\nBest model (by Macro F1): {winner.upper()}")
    
    # Save detailed results to CSV if requested
    if args.output and results:
        output_path = Path(args.output)
        all_metrics = []
        for model_name, metrics in results.items():
            for _, row in metrics['per_class_metrics'].iterrows():
                all_metrics.append({
                    'Model': model_name,
                    'Class': row['Class'],
                    'F1': row['F1'],
                })
            all_metrics.append({
                'Model': model_name,
                'Class': 'OVERALL',
                'F1': metrics['macro_f1'],
            })
            all_metrics.append({
                'Model': model_name,
                'Class': 'ACCURACY',
                'F1': metrics['accuracy'],
            })
        
        df = pd.DataFrame(all_metrics)
        df.to_csv(output_path, index=False)
        logger.info(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

