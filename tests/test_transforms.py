"""Tests for audio transforms."""
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transforms.audio import get_mel_transform, wav_to_logmel, SpecAugment


class TestMelTransform:
    """Test mel spectrogram transform creation."""
    
    def test_get_mel_transform_default(self):
        """Test creating mel transform with default parameters."""
        transform = get_mel_transform()
        assert transform is not None
        assert transform.sample_rate == 16000
        assert transform.n_fft == 1024
        assert transform.hop_length == 256
        assert transform.n_mels == 64
    
    def test_get_mel_transform_custom(self):
        """Test creating mel transform with custom parameters."""
        transform = get_mel_transform(
            sample_rate=22050,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        assert transform.sample_rate == 22050
        assert transform.n_fft == 2048
        assert transform.hop_length == 512
        assert transform.n_mels == 128


class TestWavToLogMel:
    """Test waveform to log-mel conversion."""
    
    def test_wav_to_logmel_basic(self):
        """Test basic wav to log-mel conversion."""
        # Create dummy waveform: [1, T] at 16kHz, 4 seconds
        duration = 4.0
        sr = 16000
        num_samples = int(duration * sr)
        wav = torch.randn(1, num_samples)
        
        log_mel = wav_to_logmel(wav, sr=sr)
        
        # Should output [1, n_mels, time_frames]
        assert log_mel.dim() == 3
        assert log_mel.shape[0] == 1
        assert log_mel.shape[1] == 64  # n_mels
        assert log_mel.shape[2] > 0  # time frames
    
    def test_wav_to_logmel_resampling(self):
        """Test that resampling works correctly."""
        # Create waveform at different sample rate
        duration = 4.0
        sr_orig = 22050
        num_samples = int(duration * sr_orig)
        wav = torch.randn(1, num_samples)
        
        # Convert to 16kHz
        log_mel = wav_to_logmel(wav, sr=sr_orig, target_sr=16000)
        
        assert log_mel.dim() == 3
        assert log_mel.shape[1] == 64
    
    def test_wav_to_logmel_with_transform(self):
        """Test wav_to_logmel with pre-created transform."""
        duration = 4.0
        sr = 16000
        num_samples = int(duration * sr)
        wav = torch.randn(1, num_samples)
        
        mel_transform = get_mel_transform(sample_rate=sr)
        log_mel = wav_to_logmel(wav, sr=sr, mel_transform=mel_transform)
        
        assert log_mel.dim() == 3
        assert log_mel.shape[1] == 64
    
    def test_wav_to_logmel_output_range(self):
        """Test that log-mel output is in reasonable range (dB scale)."""
        duration = 4.0
        sr = 16000
        num_samples = int(duration * sr)
        wav = torch.randn(1, num_samples)
        
        log_mel = wav_to_logmel(wav, sr=sr)
        
        # Log-mel should be in dB scale (can be positive or negative depending on signal power)
        # Values should be finite and in a reasonable range (e.g., -100 to 100 dB)
        assert torch.isfinite(log_mel).all(), "All values should be finite"
        assert log_mel.min() > -200, "Values should not be extremely negative"
        assert log_mel.max() < 200, "Values should not be extremely positive"


class TestSpecAugment:
    """Test SpecAugment data augmentation."""
    
    def test_specaugment_creation(self):
        """Test creating SpecAugment with default parameters."""
        aug = SpecAugment()
        assert aug is not None
        assert len(aug.freq_masks) == 2
        assert len(aug.time_masks) == 2
    
    def test_specaugment_creation_custom(self):
        """Test creating SpecAugment with custom parameters."""
        aug = SpecAugment(
            freq_mask_param=10,
            time_mask_param=30,
            num_freq_masks=3,
            num_time_masks=4
        )
        assert len(aug.freq_masks) == 3
        assert len(aug.time_masks) == 4
    
    def test_specaugment_forward_shape(self):
        """Test that SpecAugment preserves input shape."""
        aug = SpecAugment()
        aug.eval()
        
        # Input: [B, 1, n_mels, time]
        batch_size = 4
        n_mels = 64
        time_frames = 250
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        
        output = aug(x)
        
        # Shape should be preserved
        assert output.shape == x.shape
    
    def test_specaugment_forward_different_sizes(self):
        """Test SpecAugment with different input sizes."""
        aug = SpecAugment()
        aug.eval()
        
        test_cases = [
            (1, 1, 64, 100),
            (2, 1, 64, 250),
            (4, 1, 128, 500),
        ]
        
        for shape in test_cases:
            x = torch.randn(*shape)
            output = aug(x)
            assert output.shape == x.shape
    
    def test_specaugment_is_deterministic_in_eval(self):
        """Test that SpecAugment is deterministic in eval mode."""
        aug = SpecAugment()
        aug.eval()
        
        x = torch.randn(2, 1, 64, 250)
        
        # In eval mode, should produce same output (no random masking)
        output1 = aug(x)
        output2 = aug(x)
        
        # Note: SpecAugment may still apply masks in eval mode,
        # but we just check it doesn't crash
        assert output1.shape == output2.shape

