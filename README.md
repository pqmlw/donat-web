# Donat Classification

Website inspeksi kualitas donat (6 kelas) memakai ViT hasil fine-tuning dari Google Colab.

## 1. Ambil model dari Google Drive

Di Drive, buka folder `vit_donat_final` (hasil `trainer.save_model(...)` di Colab), unduh,
lalu salin **isinya** ke folder `model/` di proyek ini:

```
model/
├── config.json
├── model.safetensors          (atau pytorch_model.bin)
└── preprocessor_config.json
```

Urutan kelas dibaca otomatis dari `config.json` (`id2label`), jadi tidak ada yang perlu disamakan manual.
Folder lain bisa dipakai lewat `MODEL_DIR=/path/ke/vit_donat_final`.

## 2. Jalankan

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka http://localhost:5000. Saat pertama jalan, rembg mengunduh model u2net (sekitar 170 MB) otomatis.
Tab Camera butuh `localhost` atau HTTPS.

## Alur inferensi (sama dengan training)

rembg `u2net` → latar putih `(255,255,255)` → pad persegi putih → resize 224×224 (LANCZOS)
→ normalisasi mean/std dari `preprocessor_config.json` → ViT → softmax.

## Pengaturan

| Env | Default | Arti |
|---|---|---|
| `MODEL_DIR` | `./model` | lokasi model |
| `PORT` | `5000` | port server |
| `CONF_HIGH` | `0.75` | batas minimal untuk Confidence HIGH |
| `CONF_MEDIUM` | `0.50` | batas minimal untuk MEDIUM (di bawahnya LOW) |

Training memakai label smoothing 0,1, jadi probabilitas tertinggi model jarang melewati sekitar 0,92.
Itu sebabnya batas HIGH tidak dipasang di 0,9. Sesuaikan dengan hasil di data validasi kalau perlu.

## Struktur

```
app.py            backend Flask + inferensi
static/           index.html, style.css, app.js
model/            taruh model di sini
requirements.txt
```
