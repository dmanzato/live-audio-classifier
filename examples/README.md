# Examples

This directory contains example scripts and a Jupyter notebook demonstrating how to use live-audio-classifier.

## Jupyter Notebook

For an interactive tutorial, open `live-audio-classifier-tutorial.ipynb` in Jupyter:
- Load and inspect the dataset
- Create and test models
- Run inference and visualize results
- Process custom audio files

## Quick Start

### 1. Training a Model

```bash
# Set your data root
export DATA_ROOT=/path/to/UrbanSound8K

# Run training example
bash examples/train_example.sh
```

Or run directly:
```bash
python train.py \
    --data_root /path/to/UrbanSound8K \
    --train_folds 1,2,3,4,5,6,7,8,9 \
    --val_folds 10 \
    --batch_size 16 \
    --epochs 5 \
    --model smallcnn \
    --use_specaug
```

### 2. Running Inference

```bash
# On a single audio file
bash examples/inference_example.sh /path/to/your_audio.wav
```

Or run directly:
```bash
python predict.py \
    --wav /path/to/your_audio.wav \
    --data_root /path/to/UrbanSound8K \
    --model smallcnn \
    --topk 5
```

### 3. Record and Predict

Record audio from your microphone and immediately classify it:

```bash
bash examples/record_and_predict.sh
```

### 4. Live Streaming Inference

For real-time microphone input:

```bash
python scripts/stream_infer.py \
    --data_root /path/to/UrbanSound8K \
    --checkpoint artifacts/best_model.pt \
    --model smallcnn \
    --win_sec 4.0 \
    --hop_sec 0.25
```

### 5. Dataset Visualization

Browse the dataset interactively:

```bash
python scripts/vis_dataset.py \
    --data_root /path/to/UrbanSound8K \
    --folds 10 \
    --checkpoint artifacts/best_model.pt \
    --model smallcnn \
    --play_audio
```

## Environment Variables

Set these to avoid specifying paths repeatedly:

```bash
export DATA_ROOT=/path/to/UrbanSound8K
```

## Model Architectures

The project supports two model architectures:

- **SmallCNN**: Lightweight custom CNN (faster training, smaller model)
- **ResNet18**: Pre-trained ResNet adapted for audio (better accuracy, larger model)

Switch between them using the `--model` flag:
- `--model smallcnn` (default)
- `--model resnet18`

## Tips

1. **First time setup**: Make sure you have the UrbanSound8K dataset downloaded and extracted
2. **Training**: Start with SmallCNN for faster iteration, then try ResNet18 for better accuracy
3. **Inference**: The model expects 4-second audio clips at 16kHz sample rate
4. **Live streaming**: Use `--device` flag to select a specific microphone if you have multiple

## Troubleshooting

- **Import errors**: Make sure you're running from the project root with `PYTHONPATH=.`
- **Audio device errors**: On macOS, grant microphone permissions in System Settings
- **CUDA errors**: The code will fall back to CPU automatically if CUDA is unavailable

