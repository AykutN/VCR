import torch as _torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
_torch.serialization.add_safe_globals([XttsConfig, BaseDatasetConfig, XttsArgs, XttsAudioConfig])

from TTS.api import TTS
import os
import json
#import librosa
# Get device
device = "cuda" if _torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


# Load speaker wave file
# for every folder in data/


# --- Organize and clone all real samples ---

DATA_ROOT = "data"
REAL_DIR = os.path.join(DATA_ROOT, "real")
CLONED_DIR = os.path.join(DATA_ROOT, "cloned")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

for speaker_folder in os.listdir(REAL_DIR):
    speaker_path = os.path.join(REAL_DIR, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue

    # Load meta.json for sentences
    meta_path = os.path.join(speaker_path, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"[!] meta.json not found for {speaker_folder}, skipping.")
        continue
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    language = meta.get("language", "tr")
    speaker = meta.get("speaker", speaker_folder)

    # Prepare output folder
    out_speaker_dir = os.path.join(CLONED_DIR, speaker_folder)
    ensure_dir(out_speaker_dir)

    for rec in meta["recordings"]:
        real_wav = os.path.join(speaker_path, rec["file"])
        cloned_wav = os.path.join(out_speaker_dir, rec["file"])  # same filename
        if os.path.exists(cloned_wav):
            print(f"[✓] Already cloned: {cloned_wav}")
            continue
        text = rec["sentence"]
        print(f"[i] Cloning {real_wav} -> {cloned_wav} | lang={language}")
        try:
            tts.tts_to_file(text=text, speaker_wav=real_wav, language=language, file_path=cloned_wav)
            print(f"[✓] Cloned: {cloned_wav}")
        except Exception as e:
            print(f"[!] Error cloning {real_wav}: {e}")
