#!/bin/bash
# Example training script
# 
# This script demonstrates how to train a model on UrbanSound8K.
# Update DATA_ROOT with your actual dataset path.

DATA_ROOT="${DATA_ROOT:-/path/to/UrbanSound8K}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root not found: $DATA_ROOT"
    echo "Please set DATA_ROOT environment variable or update the path in this script."
    exit 1
fi

echo "Training SmallCNN model..."
python train.py \
    --data_root "$DATA_ROOT" \
    --train_folds 1,2,3,4,5,6,7,8,9 \
    --val_folds 10 \
    --batch_size 16 \
    --epochs 5 \
    --model smallcnn \
    --use_specaug \
    --lr 1e-3

echo ""
echo "Training complete! Best model saved to artifacts/best_model.pt"
echo ""
echo "To train ResNet18 instead, use:"
echo "  python train.py --data_root $DATA_ROOT --model resnet18 --epochs 5"

