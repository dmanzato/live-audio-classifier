#!/usr/bin/env python3
"""
Quick start example: Train a model and run inference.

This example demonstrates the basic workflow:
1. Train a SmallCNN model on UrbanSound8K
2. Run inference on a test audio file
3. Display predictions

Prerequisites:
- UrbanSound8K dataset downloaded and extracted
- Set DATA_ROOT environment variable or update the path below
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    # Configuration
    data_root = os.getenv("DATA_ROOT", "/path/to/UrbanSound8K")
    
    if not Path(data_root).exists():
        print(f"Error: Data root not found: {data_root}")
        print("Please set DATA_ROOT environment variable or update the path in this script.")
        return
    
    print("=" * 60)
    print("Quick Start Example")
    print("=" * 60)
    print("\nThis example shows the basic workflow.")
    print("For actual training, run:")
    print(f"  python train.py --data_root {data_root} --epochs 5 --model smallcnn")
    print("\nFor inference, run:")
    print(f"  python predict.py --wav your_audio.wav --data_root {data_root}")
    print("\nFor live streaming, run:")
    print(f"  python scripts/stream_infer.py --data_root {data_root}")
    print("=" * 60)

if __name__ == "__main__":
    main()

