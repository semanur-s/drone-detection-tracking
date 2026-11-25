# Drone Tespit ve Takibi — YOLO11 + ByteTrack

Gökyüzündeki drone'ları YOLO11 ile tespit eden, ByteTrack ile kareler arasında takip eden ve modeli gömülü cihazlar için ONNX'e çeviren uçtan uca bir sistem. Hava savunma ve İHA karşıtı sistemlerdeki "tespit → takip → gerçek zamanlı çalıştırma" zincirini küçük ölçekte kuruyor.

> **EN:** End-to-end drone detection and tracking: YOLO11n fine-tuned on the Seraphim Drone Detection Dataset, ByteTrack multi-object tracking on video, size-aware evaluation for tiny objects, and ONNX export with CPU speed benchmarks.

```
video karesi ─► YOLO11n (tespit) ─► ByteTrack (ID atama, Kalman tahmini) ─► kutu + ID + iz çizgisi
                     │
                     └─► ONNX export ─► CPU / gömülü cihazda hız ölçümü
```

## Sonuçlar

> Sonuçlar Colab'da [notebook](notebooks/drone_detection_colab.ipynb) çalıştırıldıktan sonra eklenecek.

| Metrik | Değer |
|---|---|
| Precision | – |
| Recall | – |
| mAP@0.5 | – |
| mAP@0.5:0.95 | – |
| Recall: küçük drone'lar (<32 px) | – |
| Recall: orta (32–96 px) | – |
| Recall: büyük (>96 px) | – |
| Takip: CPU'da FPS (PyTorch / ONNX) | – / – |

## Nasıl çalışıyor?

### 1. Veri
[Seraphim Drone Detection Dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset) (CC BY 4.0): 640×640, tek sınıf (drone), YOLO formatında 83 bin görüntü. Tamamı ~9 GB olduğu için `prepare_data.py` sadece ilk birkaç batch'i indirir. Eğitim verisinin %10'u doğrulama için ayrılır. Veri setinin kendi test bölümü eğitimde hiç kullanılmaz.

### 2. Tespit: YOLO11n
- COCO'da önceden eğitilmiş ağırlıklardan başlanır (transfer learning). Model kenar ve şekil gibi genel öznitelikleri zaten biliyor, sadece "drone" sınıfını öğreniyor.
- YOLO görüntüyü tek seferde ızgaralara bölüp her bölge için kutu ve güven skoru tahmin eder. Bu yüzden gerçek zamanlı çalışabilir. Klasik sliding window + HOG + SVM yaklaşımında ([klasik CV projem](https://github.com/semanur-s/classical-cv-pipeline)) her pencere ayrı ayrı sınıflandırılıyordu.
- Mozaik veri artırımı kullanılır: 4 görüntü birleştirilerek küçük nesneler farklı bağlamlarda gösterilir.

### 3. Takip: ByteTrack
Tespit her kareyi bağımsız işler. Takip ise "bu karedeki drone, önceki karedeki hangisi?" sorusunu cevaplar ve her drone'a sabit bir ID verir.
- **Kalman filtresi:** Her izin bir sonraki karedeki konumunu hız bilgisiyle tahmin eder.
- **İki aşamalı eşleştirme:** Önce yüksek güvenli tespitler izlerle eşleştirilir. Sonra düşük güvenli tespitler de kalan izlerle eşleştirilir. Küçük ya da bulanık bir drone düşük skor alsa bile izi kopmaz. ByteTrack'i diğer takipçilerden ayıran fikir budur.
- `configs/bytetrack_drone.yaml` drone'lar için uyarlandı: düşük skor eşiği 0.1 → 0.05 (küçük hedefler), iz tamponu 30 → 60 kare (drone kısa süre kaybolursa aynı ID ile geri döner).

### 4. Değerlendirme
Standart Precision, Recall ve mAP'e ek olarak `evaluate.py` gerçek drone'ları kutu alanına göre üç gruba ayırıp her grupta ayrı recall hesaplar. Uzaktaki drone'lar birkaç piksel boyutunda olduğu için toplam mAP iyi görünse bile küçük hedeflerde model zayıf kalabilir. Bu analiz o zayıflığı görünür kılar.

### 5. Dağıtım: ONNX
Model ONNX formatına çevrilir ve CPU'da PyTorch ile ONNX Runtime hızı karşılaştırılır. Raspberry Pi gibi GPU'suz bir cihaza en yakın senaryo budur.

## Çalıştırma

**Colab (önerilen):** [`notebooks/drone_detection_colab.ipynb`](notebooks/drone_detection_colab.ipynb) dosyasını Colab'da aç, T4 GPU seç ve hücreleri sırayla çalıştır.

**Yerel:**
```bash
pip install -r requirements.txt
python src/prepare_data.py --train-batches 3 --test-batches 1
python src/train.py --data data/drone.yaml --epochs 50 --device 0
W=$(cat results/best_weights.txt)
python src/evaluate.py --weights $W --data data/drone.yaml
python src/track.py --weights $W --source drone_video.mp4
python src/export_benchmark.py --weights $W --images data/drone/test/images
```

## Klasör yapısı

```
src/
├── prepare_data.py      # veri indirme, train/val/test ayırma
├── train.py             # YOLO11 ince ayar
├── evaluate.py          # test metrikleri + boyuta göre recall
├── track.py             # ByteTrack ile video takibi, iz kaydı, FPS
└── export_benchmark.py  # ONNX export + CPU hız karşılaştırması
configs/bytetrack_drone.yaml
notebooks/drone_detection_colab.ipynb
```

## Sonraki adımlar
- Küçük hedefler için daha yüksek çözünürlük (`imgsz=960`) ya da SAHI ile parçalı çıkarım
- Kuş / uçak gibi drone'a benzeyen negatif örneklerle yanlış alarmları azaltma
- TensorRT / INT8 kuantizasyon ile Jetson üzerinde ölçüm

## Kaynaklar
- Veri seti: Seraphim Drone Detection Dataset, CC BY 4.0
- [Ultralytics YOLO](https://docs.ultralytics.com) · [ByteTrack (Zhang vd., 2022)](https://github.com/ifzhang/ByteTrack)
