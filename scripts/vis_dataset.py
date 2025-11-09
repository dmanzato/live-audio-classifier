#!/usr/bin/env python3
"""
Visualize UrbanSound8K dataset files (no microphone), with audio playback and keyboard controls.
Now uses SoundFile (libsndfile) + SciPy for audio loading/resampling to avoid torchcodec.

Features
- Loads from dataset folds (same preprocessing as training: resample + pad/trim)
- Displays live-updating log-mel spectrogram
- Shows ground-truth label + model Top-K predictions
- Plays the exact audio window that produced the spectrogram
- Keyboard controls:
    Space  : pause/resume auto-advance
    Right  : next sample
    Left   : previous sample
    P      : toggle audio playback on/off
    G      : toggle spectrogram auto-gain per frame
    Q/Esc  : quit

Usage:
  export PYTHONPATH=.
  python scripts/vis_dataset.py \
    --data_root ../data/UrbanSound8K \
    --folds 10 \
    --checkpoint artifacts/best_model.pt \
    --model smallcnn \
    --spec_auto_gain \
    --sleep 0.6 \
    --play_audio
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio  # still used by your dataset; playback loader doesn't rely on it

# Ensure local imports resolve even if a PyPI package named "transforms" exists
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.urbansound8k import UrbanSound8K
from models.small_cnn import SmallCNN
from utils.device import get_device

# Playback + file I/O backends (no torchcodec needed)
try:
    import sounddevice as sd
except Exception:
    sd = None  # playback optional

import soundfile as sf
from scipy.signal import resample_poly

try:
    from torchvision.models import resnet18
except Exception:
    resnet18 = None


# -----------------------------
# Utilities
# -----------------------------
def build_model(name: str, num_classes: int):
    if name.lower() == "smallcnn":
        return SmallCNN(n_classes=num_classes)
    elif name.lower() == "resnet18":
        if resnet18 is None:
            raise RuntimeError("torchvision not available; cannot load resnet18.")
        m = resnet18(weights=None)
        # Convert first conv to handle 1 channel instead of 3
        if m.conv1.in_channels != 1:
            m.conv1 = nn.Conv2d(
                1,
                m.conv1.out_channels,
                kernel_size=m.conv1.kernel_size,
                stride=m.conv1.stride,
                padding=m.conv1.padding,
                bias=False,
            )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model name {name}")


def load_id_to_name(data_root: Path):
    """Reads UrbanSound8K.csv to map classID -> class name"""
    import pandas as pd

    meta1 = data_root / "UrbanSound8K.csv"
    meta2 = data_root / "metadata" / "UrbanSound8K.csv"

    if meta1.exists():
        meta_path = meta1
    elif meta2.exists():
        meta_path = meta2
    else:
        raise FileNotFoundError("UrbanSound8K.csv not found in data_root or data_root/metadata")

    df = pd.read_csv(meta_path)
    id_to_name = (
        df[["classID", "class"]]
        .drop_duplicates(subset=["classID"])
        .set_index("classID")["class"]
        .to_dict()
    )
    return id_to_name


def idx_to_name(dataset: UrbanSound8K, y_idx: int, id_to_name: dict):
    """
    Convert dataset's numeric label -> readable class name.

    Priority:
      1. dataset.class_ids[y_idx] -> classID -> id_to_name
      2. dataset.idx2name[y_idx] (if provided)
      3. fallback: f"class_{y_idx}"
    """
    # Try class_ids -> classID -> lookup
    if hasattr(dataset, "class_ids") and len(dataset.class_ids) > y_idx:
        class_id = int(dataset.class_ids[y_idx])  # e.g. 0..9
        if class_id in id_to_name:
            return id_to_name[class_id]

    # Try idx2name as list
    if hasattr(dataset, "idx2name") and isinstance(dataset.idx2name, list):
        if y_idx < len(dataset.idx2name):
            return dataset.idx2name[y_idx]

    # fallback
    return f"class_{y_idx}"


def load_wave_fixed(path: Path, target_sr: int, duration: float) -> torch.Tensor:
    """
    Load waveform from disk with SoundFile -> mono -> resample with SciPy -> pad/trim to fixed duration.
    Returns a tensor [1, T] at target_sr and exactly duration seconds long.
    """
    # Read file (float32)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # shape [T, C]
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)  # [T, 1]
    else:
        # already mono column
        pass
    y = data[:, 0]  # [T]

    # Resample if needed using polyphase (integer up/down)
    if sr != target_sr:
        # Up = target_sr, Down = sr (both ints)
        y = resample_poly(y, target_sr, sr).astype(np.float32)
        sr = target_sr

    # Pad/trim to exactly duration
    num_samples = int(duration * sr)
    if len(y) < num_samples:
        y = np.pad(y, (0, num_samples - len(y)))
    else:
        y = y[:num_samples]

    y = np.ascontiguousarray(y, dtype=np.float32)
    return torch.from_numpy(y)[None, :]  # [1, T]


def play_audio_blocking(wav_1xT: torch.Tensor, sr: int, out_device=None):
    """
    Play mono waveform [1, T] via sounddevice (if available). Blocks until finished.
    """
    if sd is None:
        print("[warn] sounddevice not installed; skipping playback. pip install sounddevice")
        return
    arr = wav_1xT.squeeze(0).detach().cpu().numpy()
    sd.stop(ignore_errors=True)
    sd.play(arr, sr, device=out_device)
    sd.wait()


def compute_topk(logits: torch.Tensor, k: int = 5):
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    idxs = np.argsort(-probs)[:k]
    return probs, idxs


# -----------------------------
# Viewer with keyboard controls
# -----------------------------
class Viewer:
    def __init__(self, args):
        self.args = args
        self.device = get_device()

        # Dataset
        self.data_root = Path(args.data_root)
        self.ds = UrbanSound8K(
            root=str(self.data_root),
            folds=[int(f) for f in args.folds.split(",") if f.strip()],
            target_sr=args.sr,
            duration=args.duration,
            augment=None,  # no augmentation for visualization
        )
        self.n = len(self.ds)

        # Labels mapping
        self.id_to_name = load_id_to_name(self.data_root)
        self.num_classes = len(getattr(self.ds, "class_ids", []) or getattr(self.ds, "idx2name", []) or [])
        if self.num_classes == 0:
            self.num_classes = 10

        # Model
        self.model = build_model(args.model, self.num_classes).to(self.device)
        state_dict = torch.load(args.checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Audio output device
        self.out_dev = None
        if sd is not None and args.out_device is not None:
            try:
                self.out_dev = int(args.out_device)
            except ValueError:
                devices = sd.query_devices()
                matches = [i for i, d in enumerate(devices) if args.out_device.lower() in d["name"].lower()]
                if matches:
                    self.out_dev = matches[0]

        # Matplotlib UI
        plt.ion()
        self.fig = plt.figure(figsize=(12, 7))
        # Add bottom margin to avoid long x-tick labels clipping
        self.fig.subplots_adjust(bottom=0.24)

        self.ax_spec = self.fig.add_subplot(1, 2, 1)
        self.spec_im = self.ax_spec.imshow(np.zeros((64, 10)), origin="lower", aspect="auto")
        self.ax_spec.set_title("Log-Mel Spectrogram")
        self.ax_spec.set_xlabel("Time Frames")
        self.ax_spec.set_ylabel("Mel Bins")

        self.ax_bar = self.fig.add_subplot(1, 2, 2)
        self.bars = self.ax_bar.bar(range(self.args.topk), np.zeros(self.args.topk))
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_xticks(range(self.args.topk))
        self.ax_bar.set_xticklabels([""] * self.args.topk, rotation=45, ha="right")
        self.title_txt = self.fig.suptitle("Ready")

        # State
        self.idx = 0
        self.running = True
        self.auto = True           # auto-advance
        self.play_audio = bool(args.play_audio)
        self.spec_auto_gain = bool(args.spec_auto_gain)

        # Events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # --- Key handling ---
    def on_key(self, event):
        key = (event.key or "").lower()
        if key in (" ",):             # Space = pause/resume
            self.auto = not self.auto
            self._update_status()
        elif key in ("right",):       # Next
            self.auto = False
            self.idx = min(self.n - 1, self.idx + 1)
            self.render(self.idx)
        elif key in ("left",):        # Prev
            self.auto = False
            self.idx = max(0, self.idx - 1)
            self.render(self.idx)
        elif key in ("p",):           # Toggle playback
            self.play_audio = not self.play_audio
            self._update_status()
        elif key in ("g",):           # Toggle auto gain for spectrogram
            self.spec_auto_gain = not self.spec_auto_gain
            self._update_status()
        elif key in ("q", "escape"):  # Quit
            self.running = False
        # ignore other keys

    def _update_status(self):
        status = []
        status.append("PAUSED" if not self.auto else "AUTO")
        status.append("PLAY" if self.play_audio else "MUTE")
        status.append("GAIN" if self.spec_auto_gain else "FIXED")
        self.title_txt.set_text(" | ".join(status))
        self.title_txt.set_color("black")
        self.fig.canvas.draw_idle()

    # --- Core render ---
    def render(self, i: int):
        # get dataset item
        item = self.ds[i]
        if not (isinstance(item, tuple) and len(item) >= 2):
            raise RuntimeError("Unexpected dataset item format.")
        x, y = item[0], item[1]  # x:[1, n_mels, time], y:int

        # model prediction
        xb = x.unsqueeze(0).to(self.device)  # [1,1,H,W]
        with torch.no_grad():
            logits = self.model(xb)
        probs, top_idx = compute_topk(logits, k=self.args.topk)

        # labels
        y_idx = int(y)
        gt_label = idx_to_name(self.ds, y_idx, self.id_to_name)
        top_labels = [idx_to_name(self.ds, int(j), self.id_to_name) for j in top_idx]

        # spectrogram
        arr = x[0].detach().cpu().numpy()
        self.spec_im.set_data(arr)
        if self.spec_auto_gain:
            lo, hi = np.percentile(arr, [5, 95])
            if hi > lo:
                self.spec_im.set_clim(lo, hi)

        # bars
        for bar, j in zip(self.bars, top_idx):
            bar.set_height(float(probs[j]))
        self.ax_bar.set_xticklabels(top_labels, rotation=45, ha="right")

        # title (green if correct, red otherwise)
        pred_is_correct = (top_labels[0] == gt_label)
        self.title_txt.set_text(f"GT: {gt_label} | Pred: {top_labels[0]} ({probs[top_idx[0]]:.2f})")
        self.title_txt.set_color("green" if pred_is_correct else "red")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

        # playback (SoundFile + SciPy; no torchcodec)
        if self.play_audio and sd is not None and hasattr(self.ds, "df"):
            try:
                path = Path(self.ds.df.iloc[i]["filepath"])
                wav = load_wave_fixed(path, target_sr=self.args.sr, duration=self.args.duration)
                play_audio_blocking(wav, sr=self.args.sr, out_device=self.out_dev)
            except Exception as e:
                print(f"[warn] playback failed for index {i}: {e}")

    # --- Main loop ---
    def loop(self):
        self._update_status()
        last_time = time.time()
        while self.running:
            if self.auto:
                now = time.time()
                if now - last_time >= self.args.sleep:
                    last_time = now
                    self.render(self.idx)
                    if self.idx < self.n - 1:
                        self.idx += 1
                    else:
                        self.auto = False
                        self._update_status()
                else:
                    plt.pause(0.01)
            else:
                plt.pause(0.05)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Visualize UrbanSound8K spectrogram + labels (from dataset) with keyboard controls")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--folds", type=str, default="10", help="Which folds to visualize, e.g. '10' or '1,2,3'")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--spec_auto_gain", action="store_true",
                        help="Auto-adjust spectrogram color scale per frame using 5-95th percentile")
    parser.add_argument("--sleep", type=float, default=0.6, help="Pause between samples when auto-advancing")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate to match training")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per clip (pad/trim to this)")
    parser.add_argument("--play_audio", action="store_true", help="Play the audio for each item")
    parser.add_argument("--out_device", type=str, default=None, help="Sound output device index or substring")
    args = parser.parse_args()

    viewer = Viewer(args)
    try:
        viewer.loop()
    finally:
        if sd is not None:
            try:
                sd.stop(ignore_errors=True)
            except Exception:
                pass
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
    