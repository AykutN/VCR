from batch_test import detect_deepfake
import sys

if len(sys.argv) <= 1:
    print("Usage: python classify.py /path/to/audio.wav")
    sys.exit(1)

audio_path = sys.argv[1]
result = detect_deepfake(audio_path, real_dir='data/real', threshold=0.34)

print(f"Is Fake: {result['is_fake']}")
print(f"Score: {result['score']:.4f}")
