#!/usr/bin/env python3
"""
Live streaming inference (microphone → log-mel → model) with a live spectrogram
and Top-K predictions. The Top-1 prediction is shown in the figure title.

USAGE EXAMPLES:
  # List available audio input devices
  live-audio-stream --list-devices
  # or: python scripts/stream_infer.py --list-devices

  # Minimal (uses default device, model, and checkpoint path)
  live-audio-stream \
      --data_root /path/to/UrbanSound8K \
      --checkpoint artifacts/best_model.pt

  # Pick a microphone by substring and faster refresh
  live-audio-stream \
      --data_root /path/to/UrbanSound8K \
      --checkpoint artifacts/best_model.pt \
      --device "MacBook Pro Microphone" \
      --hop_sec 0.20

  # Tweak spectrogram auto-gain percentiles
  live-audio-stream \
      --data_root /path/to/UrbanSound8K \
      --checkpoint artifacts/best_model.pt \
      --spec_pmin 3 --spec_pmax 97

TIPS
- After pip install, use the `live-audio-stream` command (no PYTHONPATH needed)
- Use --list-devices to see available microphones
- Use --device <index> or --device '<substring>' to select a specific device
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn

# Ensure local project modules resolve over any similarly named pip packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transforms.audio import get_mel_transform, wav_to_logmel  # your local transforms
from models.small_cnn import SmallCNN
from utils.device import get_device

try:
    from torchvision.models import resnet18
except Exception:
    resnet18 = None


# -----------------------------
# Model helpers
# -----------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    mname = model_name.lower()
    if mname == "smallcnn":
        return SmallCNN(n_classes=num_classes)
    elif mname == "resnet18":
        if resnet18 is None:
            raise RuntimeError("torchvision not available; cannot use resnet18")
        m = resnet18(weights=None)
        # adapt first conv to 1-channel input (log-mel)
        if m.conv1.in_channels != 1:
            m.conv1 = nn.Conv2d(
                1, m.conv1.out_channels,
                kernel_size=m.conv1.kernel_size,
                stride=m.conv1.stride,
                padding=m.conv1.padding,
                bias=False
            )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_idx2name(data_root: Path, artifacts_dir: Path) -> list[str]:
    """
    Load label names in index order. Priority:
    1) artifacts/class_map.json with key 'idx2name' (list or dict)
    2) UrbanSound8K.csv to derive classID->name and sort by classID
    """
    cm = artifacts_dir / "class_map.json"
    if cm.exists():
        with open(cm, "r") as f:
            data = json.load(f)
        if isinstance(data.get("idx2name"), list):
            return data["idx2name"]
        if isinstance(data.get("idx2name"), dict):
            items = sorted(((int(k), v) for k, v in data["idx2name"].items()), key=lambda kv: kv[0])
            return [v for _, v in items]

    # Fallback: read metadata CSV
    import pandas as pd
    meta1 = data_root / "UrbanSound8K.csv"
    meta2 = data_root / "metadata" / "UrbanSound8K.csv"
    if meta1.exists():
        meta_path = meta1
    elif meta2.exists():
        meta_path = meta2
    else:
        raise FileNotFoundError("UrbanSound8K.csv not found under data_root or data_root/metadata")
    df = pd.read_csv(meta_path)
    mapping = (
        df.loc[:, ["classID", "class"]]
          .drop_duplicates(subset=["classID"])
          .sort_values("classID")
    )
    return mapping["class"].tolist()


# -----------------------------
# Core streaming app
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Live mic → log-mel → model predictions + spectrogram")

    # Audio capture parameters
    ap.add_argument("--sr", type=int, default=16000,
                    help="Microphone sample rate (Hz). Must match model's expected transforms. [default: 16000]")
    ap.add_argument("--win_sec", type=float, default=4.0,
                    help="Rolling window length (seconds) used for inference. [default: 4.0]")
    ap.add_argument("--hop_sec", type=float, default=0.25,
                    help="UI/prediction refresh interval (seconds). [default: 0.25]")
    ap.add_argument("--device", type=str, default=None,
                    help="Microphone device index or substring to match (e.g., 'MacBook Pro Microphone').")
    ap.add_argument("--list-devices", action="store_true",
                    help="List available audio input devices and exit.")

    # Log-mel parameters (must match training)
    ap.add_argument("--n_mels", type=int, default=64, help="Number of mel bins. [default: 64]")
    ap.add_argument("--n_fft", type=int, default=1024, help="STFT FFT size. [default: 1024]")
    ap.add_argument("--hop_length", type=int, default=256, help="STFT hop length (samples). [default: 256]")

    # Model parameters
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path to UrbanSound8K root (used to resolve class names if class_map.json not present).")
    ap.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt",
                    help="Path to the trained model weights. [default: artifacts/best_model.pt]")
    ap.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18"],
                    help="Model architecture to use. [default: smallcnn]")

    # Display / smoothing
    ap.add_argument("--topk", type=int, default=5, help="How many top classes to display as bars. [default: 5]")
    ap.add_argument("--ema", type=float, default=0.6,
                    help="EMA smoothing factor for probabilities (0=no smoothing). [default: 0.6]")

    # Spectrogram scaling (stronger defaults)
    ap.add_argument("--spec_auto_gain", action="store_true", default=True,
                    help="Auto-scale spectrogram colors per refresh using percentiles (enabled by default).")
    ap.add_argument("--spec_pmin", type=float, default=5.0,
                    help="Lower percentile for auto-scaling (0..50). [default: 5.0]")
    ap.add_argument("--spec_pmax", type=float, default=95.0,
                    help="Upper percentile for auto-scaling (50..100). [default: 95.0]")
    ap.add_argument("--spec_debug", action="store_true",
                    help="Print per-frame percentile ranges to diagnose flat-color issues.")
    args = ap.parse_args()

    # Handle --list-devices flag
    if args.list_devices:
        print("Available audio input devices:")
        print("=" * 80)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']}")
                print(f"      Channels: {device['max_input_channels']}, "
                      f"Sample rate: {device['default_samplerate']} Hz")
        print("=" * 80)
        print(f"\nUse --device <index> or --device '<substring>' to select a device.")
        return

    # Device (CUDA/MPS/CPU for model)
    device = get_device()

    # Load labels
    idx2name = load_idx2name(Path(args.data_root), Path("artifacts"))
    num_classes = len(idx2name)

    # Build & load model
    model = build_model(args.model, num_classes=num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Rolling audio buffer (float32 mono samples)
    win_samples = int(args.win_sec * args.sr)
    ring = deque(maxlen=win_samples)

    # Log-mel transform (same as training)
    mel_t = get_mel_transform(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    # ---- Matplotlib UI ----
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    # Add bottom margin to prevent long x-tick labels from clipping
    fig.subplots_adjust(bottom=0.22)

    ax_spec = fig.add_subplot(1, 2, 1)
    # Initialize with a Normalize that we'll update
    init_arr = np.zeros((args.n_mels, 10), dtype=np.float32)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    spec_im = ax_spec.imshow(init_arr, aspect="auto", origin="lower", norm=norm)
    ax_spec.set_title("Live Log-Mel Spectrogram")
    ax_spec.set_xlabel("Time frames")
    ax_spec.set_ylabel("Mel bins")

    ax_bar = fig.add_subplot(1, 2, 2)
    bars = ax_bar.bar(range(args.topk), np.zeros(args.topk, dtype=np.float32))
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_xticks(range(args.topk))
    ax_bar.set_xticklabels([""] * args.topk, rotation=45, ha="right")
    title_txt = fig.suptitle("Listening…")

    # Smoothed probability vector
    ema_probs = None
    last_pred_time = 0.0

    # --- Inference helpers ---
    def predict_from_ring():
        nonlocal ema_probs, norm
        if len(ring) < win_samples:
            return None

        # Build waveform tensor [1, T]
        wav = np.asarray(ring, dtype=np.float32)
        # quick sanity: avoid NaNs from audio backend
        if not np.isfinite(wav).all():
            return None
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # [1, T]

        # Compute log-mel: [1, n_mels, time]
        log_mel = wav_to_logmel(wav_t, sr=args.sr, mel_transform=mel_t)

        # Convert to numpy and sanitize
        arr = log_mel.squeeze(0).detach().cpu().numpy()
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Update spectrogram image and dynamic normalization
        spec_im.set_data(arr)
        if args.spec_auto_gain:
            pmin = max(0.0, min(50.0, float(args.spec_pmin)))
            pmax = max(50.0, min(100.0, float(args.spec_pmax)))
            lo = float(np.percentile(arr, pmin))
            hi = float(np.percentile(arr, pmax))
            if args.spec_debug:
                print(f"[spec] p{pmin:.1f}={lo:.4f}, p{pmax:.1f}={hi:.4f}")
            if hi > lo:
                # update Normalize object (stronger than set_clim on some backends)
                norm.vmin = lo
                norm.vmax = hi
                spec_im.set_norm(norm)

        # Model inference
        x = log_mel.unsqueeze(0).to(device)  # [1, 1, H, W]
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # EMA smoothing
        if ema_probs is None:
            ema_probs = probs
        else:
            ema = float(np.clip(args.ema, 0.0, 1.0))
            ema_probs = ema * ema_probs + (1.0 - ema) * probs

        return ema_probs

    # Audio callback: append mic samples into the ring buffer
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        mono = indata.mean(axis=1).astype(np.float32) if indata.shape[1] > 1 else indata[:, 0].astype(np.float32)
        ring.extend(mono.tolist())

    # Resolve device by index or substring
    sd_device = None
    if args.device is not None:
        try:
            sd_device = int(args.device)
        except ValueError:
            devices = sd.query_devices()
            matches = [i for i, d in enumerate(devices)
                       if args.device.lower() in d["name"].lower()]
            if not matches:
                raise RuntimeError(f"No input device matches substring: {args.device!r}")
            sd_device = matches[0]

    # Start input stream
    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,            # 1-channel mono capture
        samplerate=args.sr,    # must match --sr
        device=sd_device,
        dtype="float32",
    )

    print("Streaming… Press Ctrl+C to stop.")
    with stream:
        try:
            while True:
                now = time.time()
                if now - last_pred_time >= args.hop_sec:
                    last_pred_time = now
                    probs = predict_from_ring()
                    if probs is not None:
                        # Update Top-K bar chart
                        idxs = np.argsort(-probs)[:args.topk]
                        top_labels = [idx2name[i] for i in idxs]
                        top_vals = probs[idxs]

                        for b, p in zip(bars, top_vals):
                            b.set_height(float(p))
                        ax_bar.set_xticklabels(top_labels, rotation=45, ha="right")

                        # Put Top-1 in the figure title
                        top1_idx = int(idxs[0])
                        title_txt.set_text(f"Top-1: {idx2name[top1_idx]} (p={probs[top1_idx]:.2f})")

                    # Refresh UI
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
    