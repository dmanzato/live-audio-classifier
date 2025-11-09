#!/bin/bash
# Example: Record audio and immediately run inference
#
# This script records 4 seconds of audio from your microphone,
# then runs inference on the recorded file.

DATA_ROOT="${DATA_ROOT:-/path/to/UrbanSound8K}"
OUTPUT_WAV="recorded_sample.wav"

echo "Recording 4 seconds of audio..."
python scripts/record_wav.py --out "$OUTPUT_WAV" --seconds 4 --sr 16000

if [ $? -ne 0 ]; then
    echo "Error: Recording failed"
    exit 1
fi

echo ""
echo "Running inference on recorded audio..."
python predict.py \
    --wav "$OUTPUT_WAV" \
    --data_root "$DATA_ROOT" \
    --model smallcnn \
    --checkpoint artifacts/best_model.pt \
    --topk 5

echo ""
echo "Done! Check the predictions above."

