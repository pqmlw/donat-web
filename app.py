"""
Donat Classification: backend inferensi.

Alur inferensi SAMA dengan saat training di Colab:
  1. rembg (u2net), background diganti putih (255,255,255)
  2. pad jadi persegi (putih) lalu resize 224x224 (LANCZOS)
  3. ToTensor + Normalize (mean/std dari preprocessor_config.json model)
  4. ViT hasil fine-tuning -> softmax

Jalankan:  python app.py   lalu buka http://localhost:5000
Model     : taruh isi folder `vit_donat_final` (dari Google Drive) ke folder ./model
            (atau set env MODEL_DIR=/path/ke/vit_donat_final)
"""
import base64
import io
import json
import os
import threading

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from rembg import new_session, remove
from transformers import ViTForImageClassification

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "model"))
BG = (255, 255, 255)          # harus sama dengan saat training
SIZE = 224
MAX_SIDE = 1600               # perkecil foto besar sebelum rembg (hanya demi kecepatan)
HIGH = float(os.environ.get("CONF_HIGH", 0.75))
MEDIUM = float(os.environ.get("CONF_MEDIUM", 0.50))

# Nama tampilan untuk folder kelas di dataset. Urutan indeks diambil dari config model.
LABELS = {
    "cincin_lolos":         ("Donat cincin", "Lolos", True),
    "cincin_penyok":        ("Donat cincin", "Tidak lolos: penyok", False),
    "cincin_tidaksimetris": ("Donat cincin", "Tidak lolos: tidak simetris", False),
    "isi_lolos":            ("Donat isi", "Lolos", True),
    "isi_penyok":           ("Donat isi", "Tidak lolos: penyok", False),
    "isi_bocor":            ("Donat isi", "Tidak lolos: isian bocor", False),
}

app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

_lock = threading.Lock()
_state = {"model": None, "mean": [0.5] * 3, "std": [0.5] * 3, "session": None, "error": None}


def load_model():
    cfg = os.path.join(MODEL_DIR, "config.json")
    if not os.path.isfile(cfg):
        _state["error"] = (
            f"Model belum ditemukan di '{MODEL_DIR}'. Salin isi folder vit_donat_final "
            "(config.json, model.safetensors, preprocessor_config.json) ke folder model/."
        )
        return
    try:
        model = ViTForImageClassification.from_pretrained(MODEL_DIR)
        model.eval()
        pre = os.path.join(MODEL_DIR, "preprocessor_config.json")
        if os.path.isfile(pre):
            with open(pre) as f:
                p = json.load(f)
            _state["mean"] = p.get("image_mean", _state["mean"])
            _state["std"] = p.get("image_std", _state["std"])
        _state["model"] = model
        _state["session"] = new_session("u2net")
        _state["error"] = None
    except Exception as e:  # noqa: BLE001
        _state["error"] = f"Gagal memuat model: {e}"


def preprocess(img: Image.Image):
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    # 1. hapus background -> latar putih
    out = remove(img, session=_state["session"])
    bg = Image.new("RGB", out.size, BG)
    bg.paste(out, mask=out.split()[3])

    # 2. pad persegi + resize 224
    w, h = bg.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), BG)
    canvas.paste(bg, ((s - w) // 2, (s - h) // 2))
    proc = canvas.resize((SIZE, SIZE), Image.LANCZOS)

    # 3. ToTensor + Normalize
    arr = np.asarray(proc, dtype=np.float32) / 255.0
    arr = (arr - np.array(_state["mean"], dtype=np.float32)) / np.array(_state["std"], dtype=np.float32)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x, proc


def confidence_level(p: float) -> str:
    return "HIGH" if p >= HIGH else "MEDIUM" if p >= MEDIUM else "LOW"


def describe(key: str):
    if key in LABELS:
        return LABELS[key]
    return (key.replace("_", " ").capitalize(), "", key.endswith("lolos"))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(ready=_state["model"] is not None, error=_state["error"])


@app.post("/api/predict")
def predict():
    if _state["model"] is None:
        return jsonify(error=_state["error"] or "Model belum siap."), 503
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Tidak ada gambar yang dikirim."), 400
    try:
        img = Image.open(f.stream)
        img.load()
    except Exception:  # noqa: BLE001
        return jsonify(error="File tidak bisa dibaca sebagai gambar. Gunakan JPG, PNG, atau WEBP."), 400

    with _lock:
        x, proc = preprocess(img)
        with torch.no_grad():
            probs = torch.softmax(_state["model"](pixel_values=x).logits, dim=1)[0].tolist()

    id2label = _state["model"].config.id2label
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top = ranked[0]
    key = id2label[top]
    kind, status, passed = describe(key)

    buf = io.BytesIO()
    proc.save(buf, format="JPEG", quality=90)
    return jsonify(
        key=key,
        kind=kind,
        status=status,
        passed=passed,
        probability=probs[top],
        percentage=round(probs[top] * 100, 1),
        confidence=confidence_level(probs[top]),
        ranking=[
            {"key": id2label[i], "kind": describe(id2label[i])[0], "status": describe(id2label[i])[1],
             "percentage": round(probs[i] * 100, 1)}
            for i in ranked
        ],
        processed="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
    )


load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
