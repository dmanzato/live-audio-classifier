# Minimal, shape-first demo: waveform -> log-mel -> tiny CNN
import math
import torch
from models.small_cnn import SmallCNN
from transforms.audio import wav_to_logmel, get_mel_transform

def main():
    sr = 16_000
    secs = 1.0
    t = torch.linspace(0, secs, int(sr*secs))
    freq = 1000.0
    wav = torch.sin(2 * math.pi * freq * t).unsqueeze(0)  # [1, T]

    print("Waveform shape:", wav.shape)

    mel_t = get_mel_transform(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=64)
    log_mel = wav_to_logmel(wav, sr=sr, mel_transform=mel_t)  # [1, n_mels, time]
    print("Log-mel shape:", log_mel.shape)

    x = log_mel.unsqueeze(0)  # [B=1, 1, n_mels, time]
    print("Model input shape:", x.shape)

    model = SmallCNN(n_classes=10)
    logits = model(x)
    print("Logits shape:", logits.shape)

if __name__ == "__main__":
    main()
