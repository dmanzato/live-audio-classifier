#!/usr/bin/env python3
"""
Generate a short README demo GIF (docs/demo.gif) showing:
  - log-mel spectrogram of a few clips
  - Top-K predictions (from your trained model)

Usage (examples):

  # Using UrbanSound8K fold 10 (change as needed)
  PYTHONPATH=. python scripts/gen_demo_gif.py \
    --data_root ../data/UrbanSound8K \
    --inputs UrbanSound8K/audio/fold10 \
    --checkpoint artifacts/best_model.pt \
    --model smallcnn \
    --out docs/demo.gif \
    --max_files 6

  # Or point to a directory of WAV files you have
  PYTHONPATH=. python scripts/gen_demo_gif.py \
    --inputs examples \
    --checkpoint artifacts/best_model.pt \
    --out docs/demo.gif \
    --max_files 5
"""

import argparse
from pathlib import Path
import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Make sure local imports resolve (avoid conflicting 'transforms' PyPI package)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local modules
from transforms.audio import get_mel_transform, wav_to_logmel
from utils.models import build_model
from utils.class_map import load_class_map

# File I/O (no torchcodec required)
import soundfile as sf
from scipy.signal import resample_poly


def read_mono_resampled(path: Path, target_sr: int, duration: float):
    """Read via soundfile → mono → resample with resample_poly → pad/trim to duration."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    if x.shape[1] > 1:
        x = x.mean(axis=1)
    else:
        x = x[:, 0]
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr).astype(np.float32)
        sr = target_sr
    N = int(duration * target_sr)
    if len(x) < N:
        x = np.pad(x, (0, N - len(x)))
    else:
        x = x[:N]
    return torch.from_numpy(x).unsqueeze(0)  # [1, T]


def main():
    ap = argparse.ArgumentParser("Generate README demo GIF")
    ap.add_argument("--data_root", type=str, default=".", help="UrbanSound8K root (for class names)")
    ap.add_argument("--inputs", type=str, required=True,
                    help="Directory or glob of WAV files; e.g., 'UrbanSound8K/audio/fold10' or 'examples'")
    ap.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    ap.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18"])
    ap.add_argument("--out", type=str, default="docs/demo.gif")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0, help="seconds per clip")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_files", type=int, default=6, help="Limit number of clips in demo")
    ap.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    # Use load_class_map with fallback for missing files
    try:
        idx2name = load_class_map(data_root, Path("artifacts"))
    except FileNotFoundError:
        # Fallback to generic class names if metadata not found
        idx2name = [f"class_{i}" for i in range(10)]
    num_classes = len(idx2name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Collect input wavs
    in_path = Path(args.inputs)
    if in_path.is_dir():
        wavs = sorted(glob.glob(str(in_path / "*.wav")))
    else:
        wavs = sorted(glob.glob(str(in_path)))
    if not wavs:
        raise SystemExit(f"No WAV files found under: {args.inputs}")
    wavs = wavs[: args.max_files]

    # Mel transform
    mel_t = get_mel_transform(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    # Build all frames up-front (one frame per file)
    frames = []
    topk_labels = []
    topk_probs = []

    for w in wavs:
        wav = read_mono_resampled(Path(w), target_sr=args.sr, duration=args.duration)  # [1, T]
        logmel = wav_to_logmel(wav, sr=args.sr, mel_transform=mel_t)  # [1, n_mels, time]
        x = logmel.unsqueeze(0).to(device)  # [1, 1, H, W]

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idxs = np.argsort(-probs)[: args.topk]
        labels = [idx2name[i] if i < len(idx2name) else f"class_{i}" for i in idxs]
        frames.append(logmel.squeeze(0).numpy())  # [n_mels, time]
        topk_labels.append(labels)
        topk_probs.append(probs[idxs])

    # Matplotlib figure (spectrogram + top-k bars)
    plt.ioff()
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.22)  # breathing room for x-ticks
    ax_spec = fig.add_subplot(1, 2, 1)
    ax_bar = fig.add_subplot(1, 2, 2)

    spec_im = ax_spec.imshow(np.zeros((args.n_mels, 10), dtype=np.float32),
                             origin="lower", aspect="auto")
    ax_spec.set_title("Log-Mel Spectrogram")
    ax_spec.set_xlabel("Time frames")
    ax_spec.set_ylabel("Mel bins")

    bars = ax_bar.bar(range(args.topk), np.zeros(args.topk, dtype=np.float32))
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks(range(args.topk))
    ax_bar.set_xticklabels([""] * args.topk, rotation=45, ha="right")
    title = fig.suptitle("Generating demo…")

    def update(i):
        arr = frames[i]
        labels = topk_labels[i]
        probs = topk_probs[i]
        spec_im.set_data(arr)
        # Auto gain
        lo, hi = np.percentile(arr, [5, 95])
        if hi > lo:
            spec_im.set_clim(lo, hi)
        # Bars
        for b, p in zip(bars, probs):
            b.set_height(float(p))
        ax_bar.set_xticklabels(labels, rotation=45, ha="right")
        # Title: Top-1
        title.set_text(f"Top-1: {labels[0]} (p={probs[0]:.2f})")
        return [spec_im, *bars, title]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000/args.fps), blit=False, repeat=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer=PillowWriter(fps=args.fps))
    print(f"[ok] Wrote demo GIF to: {out_path}")

if __name__ == "__main__":
    main()

