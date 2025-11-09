#!/usr/bin/env python3
"""Record audio from the microphone and save as WAV.

Examples:
  # Record 4 seconds to out.wav (mono, 16kHz)
  python scripts/record_wav.py --out out.wav --seconds 4

  # List input devices, then pick one by index
  python scripts/record_wav.py --list-devices
  python scripts/record_wav.py --out out.wav --device 3
"""
import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="recorded.wav", help="Output WAV path")
    ap.add_argument("--seconds", type=float, default=4.0, help="Duration to record (seconds)")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate")
    ap.add_argument("--channels", type=int, default=1, help="Number of channels (1=mono, 2=stereo)")
    ap.add_argument("--device", type=str, default=None, help="Input device index or substring to match")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Resolve device by index or substring
    device = None
    if args.device is not None:
        try:
            device = int(args.device)
        except ValueError:
            # Match substring in device name
            devices = sd.query_devices()
            matches = [i for i, d in enumerate(devices) if args.device.lower() in d['name'].lower()]
            if not matches:
                raise RuntimeError(f"No device matches substring: {args.device}")
            device = matches[0]

    print(f"Recording {args.seconds} seconds at {args.sr} Hz, channels={args.channels} ...")
    audio = sd.rec(int(args.seconds * args.sr), samplerate=args.sr, channels=args.channels,
                   dtype="float32", device=device)
    sd.wait()  # Wait until recording is finished
    audio = np.squeeze(audio)  # -> [T] if mono, else [T, C]

    # Save with soundfile (16-bit PCM)
    sf.write(args.out, audio, args.sr, subtype="PCM_16")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
