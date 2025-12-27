import os
import sounddevice as sd
import soundfile as sf
import time
import json

# Configuration
DATA_ROOT = "data"
REAL_DIR = os.path.join(DATA_ROOT, "real")
SENTENCES_TR = [
    "Merhaba, bu bir test cümlesidir.",
    "Bugün hava çok güzel.",
    "Yapay zeka insan hayatını kolaylaştırıyor.",
    "Lütfen bu cümleyi net bir şekilde okuyun.",
    "Ses kaydı tamamlandı."
]
SENTENCES_AR = [
    "مرحبًا، هذه جملة اختبار.",
    "الطقس جميل جدًا اليوم.",
    "الذكاء الاصطناعي يسهل حياة الإنسان.",
    "يرجى قراءة هذه الجملة بوضوح.",
    "تم الانتهاء من تسجيل الصوت."
]
SAMPLE_RATE = 22050
CHANNELS = 1

# Ensure directories exist
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Record audio
def record_sentence(out_path, duration=5):
    print(f"Kayıt başlıyor: {os.path.basename(out_path)} | Süre: {duration}s")
    input("Hazırsanız ENTER'a basın...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    for t in range(duration, 0, -1):
        print(f"Kalan: {t:2d}s", end="\r")
        time.sleep(1)
    sd.wait()
    sf.write(out_path, audio, SAMPLE_RATE)
    print(f"[✓] Kaydedildi: {out_path}")

# Main script
def main():
    speaker = input("Kullanıcı adı veya ID: ").strip().lower().replace(" ", "_")
    language = input("Dil kodu (tr/ar): ").strip().lower()
    speaker_dir = os.path.join(REAL_DIR, f"{speaker}_{language}")
    ensure_dir(speaker_dir)

    if language == "ar":
        sentences = SENTENCES_AR
    else:
        sentences = SENTENCES_TR

    meta = {"speaker": speaker, "language": language, "recordings": []}

    for idx, sentence in enumerate(sentences, 1):
        print("\nCümle:", sentence)
        fname = f"{speaker}_{language}_{idx:02d}.wav"
        out_path = os.path.join(speaker_dir, fname)
        record_sentence(out_path, duration=5)
        meta["recordings"].append({"file": fname, "sentence": sentence})

    # Save metadata
    with open(os.path.join(speaker_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nTüm kayıtlar {speaker_dir} klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
