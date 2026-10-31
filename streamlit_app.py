import base64
import io
import json
import os
import tempfile

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from ui_build import build_ui

APP = "Do or Donut"
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "model"))
BG = (255, 255, 255)  
SIZE = 224
MAX_SIDE = 1600
HIGH = float(os.environ.get("CONF_HIGH", 0.75))   
MEDIUM = float(os.environ.get("CONF_MEDIUM", 0.50))

LABELS = {
    "cincin_lolos":         ("Donat cincin", "Lolos", True),
    "cincin_penyok":        ("Donat cincin", "Tidak lolos: penyok", False),
    "cincin_tidaksimetris": ("Donat cincin", "Tidak lolos: tidak simetris", False),
    "isi_lolos":            ("Donat isi", "Lolos", True),
    "isi_penyok":           ("Donat isi", "Tidak lolos: penyok", False),
    "isi_bocor":            ("Donat isi", "Tidak lolos: isian bocor", False),
}

st.set_page_config(page_title=APP, page_icon="🍩", layout="wide", initial_sidebar_state="collapsed")

FULLSCREEN_CSS = """
<style>
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stDecoration"], footer { display: none !important; }
html, body, .stApp { background: #171012 !important; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: none !important; }
iframe[src*="/component/"] {
  position: fixed !important; top: 0; left: 0; width: 100vw !important; height: 100vh !important;
  border: 0 !important; z-index: 999999; background: transparent;
}
</style>
"""


def show_html(markup):
    fn = getattr(st, "html", None)
    if fn is not None:
        fn(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)

# model
def model_source():
    if os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        return MODEL_DIR
    mid = os.environ.get("MODEL_ID", "")
    if not mid:
        try:
            mid = st.secrets.get("MODEL_ID", "")
        except Exception: 
            mid = ""
    return mid or None


@st.cache_resource(show_spinner="Memuat model…")
def load_resources(src):
    from rembg import new_session
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(src).eval()
    mean, std = [0.5] * 3, [0.5] * 3
    try:
        if os.path.isdir(src):
            pre = os.path.join(src, "preprocessor_config.json")
        else:
            from huggingface_hub import hf_hub_download
            pre = hf_hub_download(src, "preprocessor_config.json")
        with open(pre) as f:
            p = json.load(f)
        mean, std = p.get("image_mean", mean), p.get("image_std", std)
    except Exception: 
        pass
    return model, new_session("u2net"), mean, std


def describe(key):
    return LABELS.get(key, (key.replace("_", " ").capitalize(), "", key.endswith("lolos")))


def confidence_level(p):
    return "HIGH" if p >= HIGH else "MEDIUM" if p >= MEDIUM else "LOW"


def predict(img, res):
    import torch
    from rembg import remove

    model, session, mean, std = res
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    out = remove(img, session=session)                       # 1. hapus latar
    bg = Image.new("RGB", out.size, BG)
    bg.paste(out, mask=out.split()[3])

    w, h = bg.size                                           # 2. pad persegi + resize
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), BG)
    canvas.paste(bg, ((s - w) // 2, (s - h) // 2))
    proc = canvas.resize((SIZE, SIZE), Image.LANCZOS)

    arr = np.asarray(proc, dtype=np.float32) / 255.0         # 3. normalisasi
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():                                    # 4. ViT
        probs = torch.softmax(model(pixel_values=x).logits, dim=1)[0].tolist()

    id2label = model.config.id2label
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top = ranked[0]
    key = id2label[top]
    kind, status, passed = describe(key)

    buf = io.BytesIO()
    proc.save(buf, format="JPEG", quality=90)
    return {
        "key": key, "kind": kind, "status": status, "passed": passed,
        "probability": probs[top], "percentage": round(probs[top] * 100, 1),
        "confidence": confidence_level(probs[top]),
        "ranking": [{"key": id2label[i], "kind": describe(id2label[i])[0],
                     "status": describe(id2label[i])[1], "percentage": round(probs[i] * 100, 1)}
                    for i in ranked],
        "processed": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
    }


# ---------------------------------------------------------------- komponen
@st.cache_resource
def get_component():
    out = os.path.join(tempfile.gettempdir(), "do_or_donut_ui")
    build_ui(os.path.join(BASE, "static"), os.path.join(BASE, "streamlit_bridge.js"), out)
    return components.declare_component("do_or_donut", path=out)


def load_state():
    src = model_source()
    if src is None:
        try:
            names = list(st.secrets.keys())
        except Exception:  # noqa: BLE001
            names = []
        return None, ('MODEL_ID tidak terbaca. Isi Secrets di Streamlit dengan satu baris: '
                      'MODEL_ID = "username/vit-donat", simpan, lalu Reboot app. '
                      f"Nama secret yang terbaca sekarang: {names or 'tidak ada'}.")
    try:
        return load_resources(src), None
    except Exception as e:  # noqa: BLE001
        return None, (f"Model '{src}' gagal dimuat. Pastikan repo Hugging Face berstatus Public dan berisi "
                      f"config.json, model.safetensors, preprocessor_config.json. Detail: {e}")


show_html(FULLSCREEN_CSS)
res, error = load_state()
ui = get_component()
value = ui(ready=res is not None, error=error, result=st.session_state.get("result"),
           key="ui", default=None)

# Foto baru dari komponen -> jalankan model -> kirim hasilnya kembali
if res is not None and isinstance(value, dict) and value.get("nonce") \
        and value["nonce"] != st.session_state.get("nonce"):
    nonce = value["nonce"]
    st.session_state["nonce"] = nonce
    try:
        raw = base64.b64decode(value["image"].split(",", 1)[1])
        img = Image.open(io.BytesIO(raw))
        img.load()
        out = predict(img, res)
    except Exception as e:  # noqa: BLE001
        out = {"error": f"Foto belum bisa dianalisis: {e}"}
    out["nonce"] = nonce
    st.session_state["result"] = out
    st.rerun()
