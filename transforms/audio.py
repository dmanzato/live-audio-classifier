"""Audio transforms for preprocessing and augmentation."""
from typing import Optional
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking


def get_mel_transform(
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
    power: float = 2.0
) -> MelSpectrogram:
    """
    Create a MelSpectrogram transform (power spectrogram).
    
    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Number of samples between successive frames.
        n_mels: Number of mel filterbanks.
        power: Exponent for the magnitude spectrogram (2.0 for power, 1.0 for magnitude).
    
    Returns:
        MelSpectrogram transform object.
    """
    return MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        center=True,
    )

def wav_to_logmel(
    wav: torch.Tensor,
    sr: int,
    mel_transform: Optional[MelSpectrogram] = None,
    target_sr: int = 16_000
) -> torch.Tensor:
    """
    Convert waveform to log-mel spectrogram.
    
    Args:
        wav: Input waveform tensor of shape [1, T] where T is number of samples.
        sr: Current sample rate of the waveform.
        mel_transform: Optional pre-created MelSpectrogram transform. If None, creates one.
        target_sr: Target sample rate. If different from sr, resamples the waveform.
    
    Returns:
        Log-mel spectrogram tensor of shape [1, n_mels, time_frames].
    
    Raises:
        ValueError: If input waveform has invalid shape.
    """
    if wav.dim() != 2 or wav.shape[0] != 1:
        raise ValueError(f"Expected waveform shape [1, T], got {wav.shape}")
    
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    if mel_transform is None:
        mel_transform = get_mel_transform(sample_rate=sr)
    mel = mel_transform(wav)  # [1, n_mels, time]
    to_db = AmplitudeToDB(stype="power")
    log_mel = to_db(mel)
    return log_mel


class SpecAugment(torch.nn.Module):
    """
    SpecAugment data augmentation: frequency and time masking on log-mel spectrograms.
    
    Applies random frequency and time masks to spectrograms during training to improve
    model robustness. Based on the SpecAugment paper.
    
    Args:
        freq_mask_param: Maximum width of frequency mask (in mel bins).
        time_mask_param: Maximum width of time mask (in frames).
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
    
    Example:
        >>> aug = SpecAugment(freq_mask_param=8, time_mask_param=20)
        >>> x = torch.randn(2, 1, 64, 250)  # [B, C, H, W]
        >>> x_aug = aug(x)
    """
    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 20,
                 num_freq_masks: int = 2, num_time_masks: int = 2):
        super().__init__()
        self.freq_masks = torch.nn.ModuleList(
            [FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(num_freq_masks)]
        )
        self.time_masks = torch.nn.ModuleList(
            [TimeMasking(time_mask_param=time_mask_param) for _ in range(num_time_masks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input spectrogram.
        
        Args:
            x: Input spectrogram tensor of shape [B, 1, n_mels, time_frames].
        
        Returns:
            Augmented spectrogram tensor with same shape as input.
        """
        # x: [B, 1, n_mels, time]
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x
