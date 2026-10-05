# MediaPipe ile Head-Tracking 3D Paralaks Simülasyonu

Standart bir web kamera kullanarak kullanıcının kafa hareketlerini 3 boyutlu uzayda takip eden ve ekrandaki 3D nesnelerin bakış açısına göre dinamik derinlik (off-axis projection) kazanmasını sağlayan bilgisayarlı görü uygulaması.

## 🌟 Temel Prensipler
- **MediaPipe Face Mesh:** 468 yüz referans noktası üzerinden milisaniyeler içinde göz bebekleri ve burun konumunu saptar.
- **Derinlik (Z) Kestirimi:** İki göz bebeği arasındaki piksel mesafesinin fokal lens katsayısıyla oranlanması yoluyla derinlik bulunur.
- **Off-Axis Paralaks Projeksiyonu:** Kullanıcı sağa eğildiğinde küpün sol iç yüzeyi, yukarı baktığında alt yüzeyi görünür hale gelir.

## 💻 Gereksinimler
```bash
pip install opencv-python mediapipe numpy
```

## 🚀 Çalıştırma
```bash
python head_tracking.py
```

**Geliştirici:** Berkay Bilgin ([@BERKAY081014](https://github.com/BERKAY081014))\n