
# live-audio-classifier

Real-time environmental sound classification with **PyTorch**: microphone or dataset playback → **log-mel spectrogram** → **CNN** → **live Top-K predictions** (with a clean UI and keyboard controls).

[![CI](https://img.shields.io/github/actions/workflow/status/dmanzato/live-audio-classifier/ci.yml?branch=main)](../../actions)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%AA%80-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Repository**: [https://github.com/dmanzato/live-audio-classifier](https://github.com/dmanzato/live-audio-classifier)

> **Demo**  
> ![live spectrogram + predictions](docs/demo.gif)  

---

## Project Overview

This project provides a complete pipeline for **environmental sound classification**:

- **Training**: Train CNN models (SmallCNN or ResNet18) on **UrbanSound8K** with optional data augmentation  
- **Inference**: Classify audio files with **Top-K predictions** and **spectrogram** visualization  
- **Live Streaming**: Real-time **microphone input** with **live spectrogram** and predictions  
- **Visualization**: Interactive dataset browser with ground truth **vs** predictions

### Key Features

- **Model Architectures**: SmallCNN (custom lightweight CNN) and ResNet18 (adapted for 1-channel input)  
- **Data Augmentation**: Optional SpecAugment (frequency & time masking)  
- **Live Inference**: Real-time microphone streaming with **EMA-smoothed** predictions  
- **Visualizations**: Spectrograms, confusion matrices, Top-K prediction bars  
- **Robust Audio I/O**: Uses `soundfile`/`sounddevice` (avoids torchcodec/FFmpeg dependency issues)  
- **Multi-device**: CPU, CUDA, and Apple Silicon **MPS**

### Project Structure

```
live-audio-classifier/
├── train.py              # Main training script
├── predict.py            # Single-file inference
├── demo_shapes.py        # Shape verification demo
├── models/
│   └── small_cnn.py      # CNN architecture
├── datasets/
│   └── urbansound8k.py   # Dataset loader
├── transforms/
│   └── audio.py          # Audio preprocessing & augmentation
├── scripts/
│   ├── stream_infer.py   # Live mic inference
│   ├── vis_dataset.py    # Interactive dataset viewer
│   └── record_wav.py     # Audio recording utility
├── artifacts/            # Training outputs (models, confusion matrices)
├── pred_artifacts/       # Prediction outputs (spectrograms)
└── requirements.txt      # Dependencies
```

### Core Components

**Models** (`models/small_cnn.py`):
- `SmallCNN`: Lightweight 3-layer CNN for spectrogram classification
- Input: `[B, 1, n_mels, time]` log-mel spectrograms
- Output: Class logits

**Dataset** (`datasets/urbansound8k.py`):
- `UrbanSound8K`: PyTorch Dataset loader
- Handles audio loading, resampling, padding/trimming, and log-mel conversion
- Maps classID to contiguous indices

**Transforms** (`transforms/audio.py`):
- `get_mel_transform()`: Creates MelSpectrogram transform
- `wav_to_logmel()`: Converts waveform → log-mel spectrogram
- `SpecAugment`: Frequency and time masking augmentation

---

## Quickstart

```bash
# 1) Create a venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Verify shapes end-to-end
python demo_shapes.py
```

> **CLI Commands**: After installation, you can use the entry-point commands:
> - `live-audio-train` (instead of `python train.py`)
> - `live-audio-predict` (instead of `python predict.py`)
> - `live-audio-stream` (instead of `python scripts/stream_infer.py`)
>
> These work without setting `PYTHONPATH=.`

> Want a one-liner developer flow? Use the included Makefile:
> `make setup`, `make train`, `make predict FILE=...`, `make stream DEVICE="..."`.

---

## Train on UrbanSound8K

1) Download **UrbanSound8K** and point `--data_root` to its root directory.

Expected layout:
```
UrbanSound8K/
├── audio/
│   ├── fold1/*.wav
│   ├── ...
│   └── fold10/*.wav
└── metadata/UrbanSound8K.csv
```

2) Example run:

```bash
python train.py   --data_root /path/to/UrbanSound8K   --train_folds 1,2,3,4,5,6,7,8,9   --val_folds 10   --batch_size 16   --epochs 5   --model smallcnn   --use_specaug
```

Switch to ResNet18:

```bash
python train.py   --data_root /path/to/UrbanSound8K   --train_folds 1,2,3,4,5,6,7,8,9   --val_folds 10   --batch_size 16   --epochs 5   --model resnet18
```

After each epoch you'll get:

- Macro-F1 on validation  
- A saved confusion matrix under `artifacts/confusion_matrix_epochX.png`  
- Best model checkpoint: `artifacts/best_model.pt`

---

## Inference on your own WAVs

```bash
# Uses artifacts/best_model.pt by default
python predict.py   --wav /path/to/your_sound.wav   --data_root /path/to/UrbanSound8K   --model smallcnn   --topk 5   --out_dir pred_artifacts
```

- Saves `pred_artifacts/spectrogram.png`  
- If `artifacts/class_map.json` exists it will be used; otherwise class names are read from UrbanSound8K metadata.

---

## Record audio from your mic (quick utility)

Install the extra deps (already listed in `requirements.txt`):

```bash
pip install -r requirements.txt
```

List audio devices:

```bash
python scripts/record_wav.py --list-devices
```

Record 4 seconds to WAV (mono, 16kHz):

```bash
python scripts/record_wav.py --out my_clip.wav --seconds 4 --sr 16000 --channels 1
```

Then run inference on what you recorded:

```bash
python predict.py --wav my_clip.wav   --data_root /path/to/UrbanSound8K
```

**macOS**: If you get an input-permission error, go to *System Settings → Privacy & Security → Microphone* and allow Terminal/iTerm access.

---

## Streaming Inference (Live Microphone)

Real-time microphone input with **live spectrogram** and **Top-K predictions**.

```bash
# Using the CLI command (after pip install)
live-audio-stream   --data_root /path/to/UrbanSound8K   --checkpoint artifacts/best_model.pt   --model smallcnn   --win_sec 4.0   --hop_sec 0.25   --topk 5

# Or using Python directly
python scripts/stream_infer.py   --data_root /path/to/UrbanSound8K   --checkpoint artifacts/best_model.pt   --model smallcnn   --win_sec 4.0   --hop_sec 0.25   --topk 5
```

**Features**

- Rolling window buffer (default 4 seconds)  
- Live spectrogram visualization with **auto-scaling**  
- Top-K predictions with bar chart  
- **EMA smoothing** of predictions  
- Configurable refresh rate

**Device Selection**

List available audio input devices:
```bash
live-audio-stream --list-devices
# or
python scripts/stream_infer.py --list-devices
```

Use `--device <index>` or `--device '<substring>'` to pick a specific microphone.

---

## Dataset Visualization

Interactive viewer for browsing dataset samples with predictions:

```bash
python scripts/vis_dataset.py   --data_root /path/to/UrbanSound8K   --folds 10   --checkpoint artifacts/best_model.pt   --model smallcnn   --spec_auto_gain   --play_audio
```

**Keyboard Controls**

- `Space`: Pause/resume auto-advance  
- `Left/Right`: Previous/next sample  
- `P`: Toggle audio playback  
- `G`: Toggle spectrogram auto-gain  
- `Q/Esc`: Quit

---

## Scripts Overview

### `train.py`
Main training script. Trains models on UrbanSound8K with fold-based train/val splits. Supports both SmallCNN and ResNet18 with optional SpecAugment.

### `predict.py`
Single-file inference. Classifies a WAV file and outputs Top-K predictions with probabilities. Saves spectrogram visualization.

### `scripts/stream_infer.py`
Live streaming inference from microphone. Real-time predictions with live spectrogram and prediction bars. Features EMA smoothing and auto-scaling spectrogram colors.

### `scripts/vis_dataset.py`
Interactive dataset browser. Shows ground truth vs predictions, spectrograms, and supports audio playback with keyboard navigation.

### `scripts/record_wav.py`
Audio recording utility. Records from microphone and saves as WAV file (mono, 16kHz by default).

### `demo_shapes.py`
Minimal shape verification script. Tests the pipeline end-to-end: waveform → log-mel → model to verify tensor shapes.

---

## Examples

See the [examples/](examples/) directory for:
- Training scripts with different configurations
- Inference examples
- Record-and-predict workflows
- Jupyter notebook tutorials
- Complete usage documentation

---

## Why this repo?

Most UrbanSound8K examples stop at “train offline & classify WAVs.”  
This project also provides **real-time microphone streaming**, **live spectrogram visualization**, **keyboard controls**, and **dataset playback with synchronized audio**, all in a minimal, hackable codebase.

```
audio → STFT → Mel → Log → CNN → Top-K + UI
```

---

## Testing

Run the test suite:

```bash
pytest tests/
```

For a coverage report:

```bash
pytest tests/ --cov=. --cov-report=html
```

See `tests/README.md` for more details.

---

## License

[MIT](LICENSE)
