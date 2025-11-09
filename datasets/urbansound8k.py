import os
from typing import List, Optional, Tuple
import pandas as pd
import torch
import torchaudio
import soundfile as sf

from transforms.audio import get_mel_transform, wav_to_logmel

class UrbanSound8K(torch.utils.data.Dataset):
    """UrbanSound8K dataset loader.
    Expects directory layout like:
      root/
        UrbanSound8K.csv            (metadata)  OR metadata/UrbanSound8K.csv
        audio/
          fold1/*.wav
          fold2/*.wav
          ...
    Args:
        root: dataset root directory
        folds: list of fold integers to include (1..10)
        target_sr: resample audio to this sample rate
        duration: seconds to which clips are trimmed/padded
        n_mels, n_fft, hop_length: spectrogram params
        augment: optional callable (e.g., SpecAugment) applied on spectrogram tensors [B,1,H,W] or per-sample [1,1,H,W]
    """
    def __init__(
        self,
        root: str,
        folds: List[int],
        target_sr: int = 16_000,
        duration: float = 4.0,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        augment = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.folds = set(int(f) for f in folds)
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

        # find metadata CSV
        csv1 = os.path.join(root, "UrbanSound8K.csv")
        csv2 = os.path.join(root, "metadata", "UrbanSound8K.csv")
        if os.path.isfile(csv1):
            meta_path = csv1
        elif os.path.isfile(csv2):
            meta_path = csv2
        else:
            raise FileNotFoundError("UrbanSound8K.csv not found under root/ or root/metadata/")

        df = pd.read_csv(meta_path)
        # Expected columns: slice_file_name, fold, classID, class, start, end, ...
        df = df[df["fold"].isin(self.folds)].copy()
        # Build absolute file paths
        audio_root = os.path.join(root, "audio")
        df["filepath"] = df.apply(
            lambda r: os.path.join(audio_root, f"fold{int(r['fold'])}", str(r["slice_file_name"])),
            axis=1
        )
        # Filter only existing files
        df = df[df["filepath"].apply(os.path.isfile)].reset_index(drop=True)

        self.df = df
        self.class_ids = sorted(self.df["classID"].unique().tolist())
        # Map classID to 0..C-1 contiguous
        self.id2idx = {cid: i for i, cid in enumerate(self.class_ids)}
        self.idx2name = {self.id2idx[cid]: name for cid, name in zip(self.df["classID"], self.df["class"])}

        # Prepare mel transform
        self.mel_t = get_mel_transform(
            sample_rate=self.target_sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )

        # Pre-compute number of samples per clip
        self.num_samples = int(self.duration * self.target_sr)

    def __len__(self) -> int:
        return len(self.df)

    def _load_wave_fixed(self, path: str) -> torch.Tensor:
        # Load using soundfile directly to avoid torchcodec/FFmpeg dependency issues
        try:
            # Try using soundfile directly (more reliable)
            wav, sr = sf.read(path, always_2d=True, dtype='float32')
            wav = torch.from_numpy(wav).T  # Convert to [C, T] format
        except Exception:
            # Fallback to torchaudio with soundfile backend
            wav, sr = torchaudio.load(path, backend="soundfile")  # [C, T]
        # To mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.target_sr)
            sr = self.target_sr
        # Pad/trim to fixed duration
        T = wav.size(1)
        if T < self.num_samples:
            pad = self.num_samples - T
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif T > self.num_samples:
            wav = wav[:, :self.num_samples]
        return wav  # [1, num_samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = row["filepath"]
        class_id = int(row["classID"])
        y = self.id2idx[class_id]

        wav = self._load_wave_fixed(path)  # [1, T]
        log_mel = wav_to_logmel(wav, sr=self.target_sr, mel_transform=self.mel_t)  # [1, n_mels, time]

        # Add channel dimension for Conv2d: [1, 1, n_mels, time]
        x = log_mel.unsqueeze(0)

        # Optional per-sample augment (e.g., SpecAugment expects [B,1,H,W])
        if self.augment is not None:
            x = self.augment(x)
        # Remove batch dim back: [1, n_mels, time]
        x = x.squeeze(0)

        return x, y

def folds_split(train_folds: List[int], val_folds: List[int]):
    return list(train_folds), list(val_folds)
