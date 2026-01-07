# Deepfake (Klonlanmış) Ses Tespiti – Teorik Açıklama ve Projede Ne Yaptık? (TR)

Bu metin, proje deposunda (VCR) geliştirilen **klonlanmış ses (deepfake voice)** tespit sisteminin teorik temellerini, kullandığımız sinyal işleme kavramlarını ve uygulama akışını anlatan açıklayıcı bir rapordur. Sinyal ve Görüntü İşleme sınavı için hazırlanmıştır.

---

## 1) Problem Tanımı ve Amaç

**Amaç:** Verilen bir ses dosyasının **gerçek (real)** mi yoksa **klonlanmış/yapay (cloned, deepfake)** mı olduğunu tespit etmek.

**Neden zor?** Deepfake sesler zaman alanında insan konuşmasına çok benzer; fakat genellikle:
- Spektral ayrıntılarda (enerji dağılımı, yüksek frekans davranışı),
- Zamansal dinamiklerde (tonlama/prosodi, geçişler),
- Gürültü-tonu/tonal-gürültü dengesinde
küçük ama yakalanabilir farklar bırakır.

Sistemimiz bu farkları **özellik çıkarımı (feature extraction)** ile sayısallaştırıp **skor** üretir ve bir **eşik (threshold)** ile karar verir.

---

## 2) Veri Yapısı ve Deney Senaryosu

Veriler şu şekilde düzenlenir:
- `data/real/<speaker>/*.wav` : Gerçek ses kayıtları (referans)
- `data/cloned/<speaker>/*.wav` : Aynı konuşmacıya ait klonlanmış kayıtlar

Temel fikir:
1. **Gerçek seslerden** bir “referans dağılımı” oluşturmak.
2. Test edilecek sesin özelliklerini çıkarıp bu referansla karşılaştırmak.
3. Fark büyükse **Fake**, küçükse **Real** demek.

---

## 3) Ön-İşleme (Preprocessing)

Projede tüm sesler `librosa.load(..., sr=22050)` ile yüklenir.

Bu adımın amacı:
- Farklı örnekleme oranına sahip dosyaları **tek bir örnekleme oranında** standardize etmek,
- Özellik çıkarımının (MFCC, spektral özellikler) tutarlı çalışmasını sağlamak.

Teorik kavramlar:
- **Sampling rate (örnekleme frekansı)**
- **Yeniden örnekleme (resampling)**
- (Kısa bilgi) **Nyquist** ve aliasing riski

---

## 4) Zaman–Frekans Analizi ve Konuşma Sinyali

Konuşma sinyali **durağan (stationary)** değildir; zaman içinde hızlı değişir. Bu yüzden analiz çoğunlukla kısa pencereli yapılır.

Kullandığımız birçok özellik (MFCC, spektral centroid/rolloff/bandwidth vb.) pratikte **kısa-zamanlı spektral analiz** (STFT mantığı) üzerinden elde edilir.

STFT tarafındaki temel kavramlar:
- **Pencereleme (windowing)** ve **frame** kavramı
- **n_fft**: frekans çözünürlüğü
- **hop_length**: zaman adımı
- Zaman–frekans çözünürlüğü değiş tokuşu

Projede kullanılan tipik parametreler:
- `n_fft = 2048`
- `hop_length = 512`

---

## 5) Özellik Çıkarımı (Feature Extraction) – Repo Mantığı

Özellik çıkarımı ağırlıklı olarak `batch_test.py` içindeki `extract_all_features()` fonksiyonuyla yapılır.

### 5.1 MFCC (Mel-Frequency Cepstral Coefficients)
- **Ne?** Konuşmanın spektral zarfını (vokal trakt bilgisi) sıkıştırılmış şekilde temsil eden katsayılar.
- Projede **13 MFCC** çıkarılır (`n_mfcc=13`).

Teorik akış (özet):
1. Sinyal kısa parçalara bölünür (frame)
2. Spektrum elde edilir
3. **Mel filterbank** uygulanır
4. Log ölçek alınır
5. DCT ile cepstral katsayılar (MFCC) üretilir

### 5.2 Delta ve Delta-Delta
- **Delta:** MFCC’nin zaman içindeki 1. türevi (değişim hızı)
- **Delta-Delta:** 2. türev (ivmelenme)

Amaç: Konuşmanın **dinamik özelliklerini** yakalamak. Deepfake üretiminde bu dinamikler bazen daha “mekanik/tekdüze” olabilir.

### 5.3 Spektral (Fourier tabanlı) özellikler
Projede çıkarılan temel spektral özellikler:
- **Spectral centroid**: spektrumun “ağırlık merkezi” (sesin parlaklığı)
- **Spectral rolloff**: enerjinin büyük kısmının toplandığı üst frekans sınırı
- **Zero crossing rate (ZCR)**: işaretin sıfır çizgisini geçme sıklığı (voicing/gürültüsellik)
- **Spectral bandwidth**: spektral yayılım genişliği
- **RMS energy**: enerji
- **Spectral flatness**: tonal mı gürültümsü mü ölçüsü

(Not: Kodda `mel_spectrogram` da hesaplanır; ancak sınıflandırma vektörüne asıl eklenenler yukarıdaki ana spektral ölçütlerdir.)

---

## 6) İstatistiksel Özetleme (Sabit Boyutlu Feature Vektörü)

Konuşma özellikleri frame bazlı çıktığı için (ör. MFCC matrisi: 13 × T), her dosyanın uzunluğu değişebilir.

Bunu sabitlemek için her özellikten istatistiksel özetler çıkarıyoruz:
- **mean (ortalama)**
- **std (standart sapma)**
- **skewness (çarpıklık)**
- **kurtosis (basıklık)**
(Bazı yerlerde min/max/median da hesaplanır.)

Sonuç: Her dosya için tek bir **feature vector** elde edilir.
Bu vektör `flatten_statistical_features()` ile birleştirilir.

---

## 7) Kural Tabanlı (Rule-Based) Tespit – Ana Sistem

Kural tabanlı tespit `detect_deepfake()` (dosya: `batch_test.py`) içinde yapılır.
Tek bir “ipuç” yerine 3 farklı istatistiksel yaklaşım birleştirilir.

### 7.1 Mesafe Tabanlı Benzerlik (Euclidean Distance)
- Test vektörü ile her referans gerçek vektör arasındaki **Öklid mesafesi** hesaplanır.
- Ortalama mesafe büyüdükçe “fake olma” ihtimali artar.

### 7.2 Referans Dağılıma Göre Eşik İhlali (2σ / 3σ)
- Referans real vektörlerden her feature için:
  - `μ ± 2σ` ve `μ ± 3σ` sınırları çıkarılır.
- Test vektöründe kaç feature bu aralıkların dışına taşıyor?
  - Taşma sayısı/ oranı arttıkça “anomali” artar.

### 7.3 Z-Score ile İstatistiksel Anormallik
- Her feature için `z = |(x - μ)/σ|` hesaplanır.
- `z > 2` ve `z > 3` olan feature sayısı yüksekse skor artar.

### 7.4 Hibrit Skor Birleştirme
Bu üç bileşen ağırlıklı ortalama ile tek skora çevrilir:
- distance ağırlığı: **0.3**
- threshold ağırlığı: **0.4**
- statistical ağırlığı: **0.3**

Skor: `0–1` arası.
Karar: `score >= threshold` ise **Fake**, aksi halde **Real**.

---

## 8) Eşik (Threshold) Seçimi ve İyileştirme

Sistemin başarısı büyük ölçüde **threshold** seçimine bağlıdır.

Projede yapılan kritik iyileştirme:
- Başlangıçta `threshold = 0.5` ile klonları yakalama zayıftı.
- Skor dağılımı analizi sonrası `threshold = 0.34` seçildi.

Bu, pratikte bir tür **kalibrasyon / karar sınırı optimizasyonu** problemidir.

---

## 9) ML Tabanlı Tespit (Supervised Learning)

Repo ayrıca ML tabanlı bir yöntem de içerir (`ml_detector.py`).

Akış:
1. Aynı feature vector çıkarılır.
2. Önceden eğitilmiş modeller yüklenir:
   - **Logistic Regression**
   - **SVM**
3. `scaler.pkl` ile özellikler standardize edilir.
4. Her model “fake olasılığı” üretir.
5. İki skorun ortalaması alınır:
   - `combined_score = (lr_score + svm_score)/2`

Karar eşiği: `0.5`.

Teorik kavramlar:
- sınıflandırma, olasılık (predict_proba), standardizasyon, overfitting riski

---

## 10) Hibrit Sistem (Rule + ML Skor Füzyonu)

`hybrid_detector.py` iki yaklaşımı bir araya getirir:
- `rule_score` (kural tabanlı)
- `ml_score` (LR+SVM)

Ağırlıklı birleştirme:
- `hybrid_score = rule_weight * rule_score + ml_weight * ml_score`
- Varsayılan: 0.5 / 0.5

Karar eşiği: `0.5`.

Bu yaklaşım teoride **ensemble / score fusion** mantığıdır.

---

## 11) Değerlendirme (Metrikler)

Toplu test akışı `batch_test.py` ile tüm real ve cloned dosyalar üzerinde denenebilir.

Sınavda bahsedilebilecek metrikler:
- **Confusion matrix**
- Accuracy
- Precision / Recall
- F1-score

Ek olarak; projenin güçlü yanı, sadece accuracy değil **skor dağılımlarını** inceleyip doğru threshold seçmesidir.

---

## 12) Sınırlılıklar ve Gelecek Çalışmalar

**Sınırlılıklar:**
- Referans real veri azsa μ/σ tahmini kararsız olur.
- Farklı mikrofon/ortam/sıkıştırma (domain shift) performansı düşürebilir.
- Deepfake üreticileri geliştikçe özellik farkları küçülebilir.

**Geliştirme fikirleri:**
- Konuşmacı-bağımsız değerlendirme (speaker leakage önleme)
- Veri artırma (noise/reverb) ile genelleme
- Spektrogram tabanlı CNN/CRNN ya da self-supervised embedding (wav2vec2) ile daha güçlü modeller

---

## Kısa Özet (Sözlü anlatım için 30 saniye)

Bu projede ses dosyalarından MFCC, delta/delta-delta ve spektral (centroid, rolloff, ZCR, bandwidth, RMS, flatness) özelliklerini çıkarıp bunları istatistiksel olarak özetleyerek sabit boyutlu vektöre çeviriyoruz. Gerçek seslerden referans dağılım oluşturup test sesini hem Öklid mesafesiyle hem de 2σ/3σ eşik ihlalleri ve z-score anormalliğiyle karşılaştırıyoruz. Bu üç bileşeni ağırlıklı birleştirip 0–1 arası skor üretiyoruz ve threshold ile fake/real karar veriyoruz. Ayrıca aynı feature vektörünü Logistic Regression + SVM ile eğitip ML skoru elde ediyor, istenirse iki yaklaşımı hibrit şekilde birleştiriyoruz.
