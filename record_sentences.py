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
    "Ses kaydı tamamlandı.",
    "Teknoloji her geçen gün gelişiyor ve değişiyor.",
    "Bilgisayar bilimleri modern dünyanın temel taşlarından biridir.",
    "Doğal dil işleme yapay zekanın önemli bir dalıdır.",
    "Ses tanıma sistemleri günlük hayatımızda yaygın olarak kullanılıyor.",
    "Makine öğrenmesi algoritmaları büyük veri setlerinden öğrenir.",
    "Derin öğrenme sinir ağları ile gerçekleştirilen bir yöntemdir.",
    "Sesli asistanlar evlerimizde ve ofislerimizde yardımcı oluyor.",
    "Dijital dönüşüm tüm sektörleri etkiliyor ve değiştiriyor.",
    "Yapay zeka uygulamaları sağlık sektöründe devrim yaratıyor.",
    "Ses kalitesi iletişimde çok önemli bir faktördür.",
    "Bilimsel araştırmalar insanlığın ilerlemesine katkı sağlıyor.",
    "Teknolojik yenilikler yaşam kalitemizi artırıyor.",
    "Ses işleme teknikleri müzik ve film endüstrisinde kullanılıyor.",
    "Yapay zeka ve robotik geleceğin mesleklerini şekillendiriyor.",
    "Dijital çağda bilgiye erişim çok daha kolay hale geldi."
]
SENTENCES_AR = [
    "مرحبًا، هذه جملة اختبار.",
    "الطقس جميل جدًا اليوم.",
    "الذكاء الاصطناعي يسهل حياة الإنسان.",
    "يرجى قراءة هذه الجملة بوضوح.",
    "تم الانتهاء من تسجيل الصوت.",
    "التكنولوجيا تتطور وتتغير كل يوم.",
    "علوم الحاسوب هي إحدى الركائز الأساسية للعالم الحديث.",
    "معالجة اللغة الطبيعية هي فرع مهم من الذكاء الاصطناعي.",
    "أنظمة التعرف على الصوت تُستخدم على نطاق واسع في حياتنا اليومية.",
    "خوارزميات التعلم الآلي تتعلم من مجموعات البيانات الكبيرة.",
    "التعلم العميق هو طريقة تُنفذ باستخدام الشبكات العصبية.",
    "المساعدات الصوتية تساعد في منازلنا ومكاتبنا.",
    "التحول الرقمي يؤثر ويغير جميع القطاعات.",
    "تطبيقات الذكاء الاصطناعي تحدث ثورة في قطاع الصحة.",
    "جودة الصوت عامل مهم جدًا في التواصل.",
    "البحوث العلمية تساهم في تقدم البشرية.",
    "الابتكارات التكنولوجية تحسن نوعية حياتنا.",
    "تقنيات معالجة الصوت تُستخدم في صناعة الموسيقى والأفلام.",
    "الذكاء الاصطناعي والروبوتات تشكل مهن المستقبل.",
    "في العصر الرقمي، أصبح الوصول إلى المعلومات أسهل بكثير."
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
