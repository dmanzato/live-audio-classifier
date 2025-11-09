import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from transforms.audio import get_mel_transform, wav_to_logmel
from models.small_cnn import SmallCNN
from utils.device import get_device
try:
    from torchvision.models import resnet18
except Exception:
    resnet18 = None

def build_model(model_name: str, num_classes: int):
    if model_name.lower() == "smallcnn":
        return SmallCNN(n_classes=num_classes)
    elif model_name.lower() == "resnet18":
        if resnet18 is None:
            raise RuntimeError("torchvision not available; cannot use resnet18")
        m = resnet18(weights=None)
        if m.conv1.in_channels != 1:
            m.conv1 = nn.Conv2d(1, m.conv1.out_channels, kernel_size=m.conv1.kernel_size,
                                stride=m.conv1.stride, padding=m.conv1.padding, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")

def load_class_map(data_root: Path, artifacts_dir: Path):
    cm1 = artifacts_dir / "class_map.json"
    if cm1.exists():
        with open(cm1, "r") as f:
            data = json.load(f)
        if isinstance(data.get("idx2name"), list):
            return data["idx2name"]
        elif isinstance(data.get("idx2name"), dict):
            idx2name = [v for k, v in sorted(data["idx2name"].items(), key=lambda kv: int(kv[0]))]
            return idx2name

    import pandas as pd
    meta1 = data_root / "UrbanSound8K.csv"
    meta2 = data_root / "metadata" / "UrbanSound8K.csv"
    if meta1.exists():
        meta_path = meta1
    elif meta2.exists():
        meta_path = meta2
    else:
        raise FileNotFoundError("Could not find UrbanSound8K.csv under data_root or data_root/metadata.")
    df = pd.read_csv(meta_path)
    id_to_name = df.drop_duplicates(subset=["classID"])[["classID", "class"]].sort_values("classID")
    idx2name = id_to_name["class"].tolist()
    return idx2name

def topk_probs(logits: torch.Tensor, k: int = 5):
    probs = torch.softmax(logits, dim=-1)
    p, i = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)
    return p[0].detach().cpu().numpy(), i[0].detach().cpu().numpy()

def save_spectrogram_image(log_mel: torch.Tensor, out_path: Path):
    arr = log_mel.squeeze(0).detach().cpu().numpy()
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(arr, aspect='auto', origin='lower')
    plt.title('Log-Mel Spectrogram')
    plt.xlabel('Time frames')
    plt.ylabel('Mel bins')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Predict UrbanSound8K class for a WAV file.")
    ap.add_argument("--wav", type=str, required=True, help="Path to a mono WAV (any sr; will be resampled).")
    ap.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt", help="Model weights .pt file.")
    ap.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18"], help="Model type.")
    ap.add_argument("--data_root", type=str, required=True, help="UrbanSound8K root (for class names).")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate for preprocessing.")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--duration", type=float, default=4.0, help="Seconds; clip will be padded/trimmed.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="pred_artifacts")
    args = ap.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    idx2name = load_class_map(Path(args.data_root), Path("artifacts"))
    num_classes = len(idx2name)

    model = build_model(args.model, num_classes=num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Load using soundfile directly to avoid torchcodec/FFmpeg dependency issues
    wav, sr = sf.read(args.wav, always_2d=True, dtype='float32')
    wav = torch.from_numpy(wav).T  # Convert to [C, T] format
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != args.sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=args.sr)
        sr = args.sr
    num_samples = int(args.duration * sr)
    T = wav.size(1)
    if T < num_samples:
        wav = torch.nn.functional.pad(wav, (0, num_samples - T))
    elif T > num_samples:
        wav = wav[:, :num_samples]

    mel_t = get_mel_transform(sample_rate=sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
    log_mel = wav_to_logmel(wav, sr=sr, mel_transform=mel_t)

    spec_path = out_dir / "spectrogram.png"
    save_spectrogram_image(log_mel, spec_path)

    x = log_mel.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)

    probs, indices = topk_probs(logits, k=args.topk)

    print("\nTop predictions:")
    for rank, (p, idx) in enumerate(zip(probs, indices), start=1):
        name = idx2name[int(idx)] if int(idx) < len(idx2name) else f"class_{int(idx)}"
        print(f"{rank}. {name:20s}  prob={p:.3f} (idx={int(idx)})")

    print(f"Saved spectrogram image to: {spec_path}")

if __name__ == "__main__":
    main()
