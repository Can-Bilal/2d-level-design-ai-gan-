# Unity Level Generator GAN

Bu proje, Unity oyunları için otomatik level tasarımı yapan bir Generative Adversarial Network (GAN) uygulamasıdır.

## Proje Yapısı

- `main.py`: Ana uygulama dosyası
- `models/`: Model mimarileri
  - `generator.py`: Level üreten model
  - `discriminator.py`: Level'ların gerçekçiliğini değerlendiren model
- `data/`: Veri işleme
  - `data_loader.py`: Veri yükleme ve işleme fonksiyonları
- `utils/`: Yardımcı araçlar
  - `visualize.py`: Level görselleştirme araçları
- `train.py`: Model eğitim kodu

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Modeli eğitmek için:
```bash
python train.py
```

## Level Formatı

Level'lar 32x16 boyutunda bir grid olarak temsil edilir:
- 0: Boş alan
- 1: Platform
- 2: Coin/Ödül
- 3: Engel
- 4: Başlangıç noktası
- 5: Bitiş noktası

## Unity Entegrasyonu

Üretilen level'lar PNG formatında `outputs/` klasörüne kaydedilir. Bu görüntüler Unity'de prefab'lara dönüştürülerek kullanılabilir.
