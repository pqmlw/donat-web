"""
Donat Classification (versi Streamlit).

Alur inferensi SAMA dengan training di Colab:
  rembg (u2net) -> latar putih -> pad persegi -> resize 224x224 -> normalisasi -> ViT

Jalankan:  streamlit run streamlit_app.py
Model     : isi folder vit_donat_final (config.json, model.safetensors,
            preprocessor_config.json) taruh di ./model  (atau set env MODEL_DIR)
"""
import json
import os

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from rembg import new_session, remove
from transformers import ViTForImageClassification

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "model"))
BG = (255, 255, 255)          # harus sama dengan saat training
SIZE = 224
MAX_SIDE = 1600
HIGH = float(os.environ.get("CONF_HIGH", 0.75))     # sesuaikan dengan data validasi
MEDIUM = float(os.environ.get("CONF_MEDIUM", 0.50))

LABELS = {
    "cincin_lolos":         ("Donat cincin", "Lolos", True),
    "cincin_penyok":        ("Donat cincin", "Tidak lolos: penyok", False),
    "cincin_tidaksimetris": ("Donat cincin", "Tidak lolos: tidak simetris", False),
    "isi_lolos":            ("Donat isi", "Lolos", True),
    "isi_penyok":           ("Donat isi", "Tidak lolos: penyok", False),
    "isi_bocor":            ("Donat isi", "Tidak lolos: isian bocor", False),
}
PASS_C, FAIL_C = "#1b7f55", "#c23b22"

st.set_page_config(page_title="Donat Classification", layout="wide")


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Memuat model…")
def load_resources():
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        return None
    model = ViTForImageClassification.from_pretrained(MODEL_DIR).eval()
    mean, std = [0.5] * 3, [0.5] * 3
    pre = os.path.join(MODEL_DIR, "preprocessor_config.json")
    if os.path.isfile(pre):
        with open(pre) as f:
            p = json.load(f)
        mean, std = p.get("image_mean", mean), p.get("image_std", std)
    return model, new_session("u2net"), mean, std


def describe(key):
    return LABELS.get(key, (key.replace("_", " ").capitalize(), "", key.endswith("lolos")))


def confidence_level(p):
    return "HIGH" if p >= HIGH else "MEDIUM" if p >= MEDIUM else "LOW"


def predict(img, res):
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
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    key = id2label[order[0]]
    kind, status, passed = describe(key)
    return {
        "kind": kind, "status": status, "passed": passed,
        "prob": probs[order[0]], "conf": confidence_level(probs[order[0]]),
        "ranking": [(describe(id2label[i])[0], describe(id2label[i])[1], probs[i] * 100) for i in order],
        "processed": proc,
    }


def ring_svg(prob, passed):
    c = 2 * np.pi * 46
    color = PASS_C if passed else FAIL_C
    icon = ("M40 62 L54 76 L82 44" if passed else "M44 44 L76 76 M76 44 L44 76")
    return f"""
    <svg width="120" height="120" viewBox="0 0 120 120">
      <g transform="rotate(-90 60 60)">
        <circle cx="60" cy="60" r="46" fill="none" stroke="#e1dde2" stroke-width="15"/>
        <circle cx="60" cy="60" r="46" fill="none" stroke="{color}" stroke-width="15"
                stroke-dasharray="{c * prob:.1f} {c:.1f}"/>
      </g>
      <path d="{icon}" fill="none" stroke="{color}" stroke-width="9"
            stroke-linecap="round" stroke-linejoin="round"/>
    </svg>"""


# ---------------------------------------------------------------- halaman
def page_home():
    res = load_resources()
    if res is None:
        st.error(f"Model belum ditemukan di '{MODEL_DIR}'. Salin config.json, model.safetensors, "
                 "dan preprocessor_config.json dari vit_donat_final ke folder model/.")
        return

    st.session_state.setdefault("k", 0)
    st.session_state.setdefault("result", None)
    k = st.session_state.k

    if st.session_state.result is None:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            tab_up, tab_cam = st.tabs(["Upload", "Camera"])
            with tab_up:
                up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"], key=f"up{k}")
            with tab_cam:
                cam = st.camera_input("Take photo", key=f"cam{k}")
            f = up or cam
            if f is not None:
                img = Image.open(f)
                img.load()
                with st.spinner("Menganalisis foto…"):
                    r = predict(img.copy(), res)
                st.session_state.result = {"img": img, **r}
                st.rerun()
        return

    r = st.session_state.result
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(r["img"], width=380)
    with right:
        st.caption(r["kind"])
        color = PASS_C if r["passed"] else FAIL_C
        st.markdown(f"<h1 style='color:{color};margin:0 0 8px 0'>{r['status'] or ('Lolos' if r['passed'] else 'Tidak lolos')}</h1>",
                    unsafe_allow_html=True)
        g, t = st.columns([1, 2])
        with g:
            components.html(ring_svg(r["prob"], r["passed"]), height=130)
        with t:
            st.markdown(f"**Percentage:** {r['prob'] * 100:.0f}%  \n**Confidence:** {r['conf']}")
        if r["conf"] == "LOW":
            st.warning("Model ragu dengan foto ini. Periksa donat secara manual atau coba foto lain.")
        if st.button("Try another image"):
            st.session_state.result = None
            st.session_state.k += 1
            st.rerun()
        with st.expander("Peluang tiap kelas"):
            st.table([{"Kelas": f"{kind}, {status.lower()}" if status else kind, "Peluang (%)": round(p, 1)}
                      for kind, status, p in r["ranking"]])
            st.image(r["processed"], width=112,
                     caption="Gambar yang dilihat model (latar dihapus, 224×224). "
                             "Pastikan bagian cacat tidak ikut terhapus.")


def page_tutorial():
    st.title("How to use?")
    st.markdown("""
1. **Pilih sumber foto.** Tab *Upload* untuk foto dari perangkat, tab *Camera* untuk memotret langsung.
2. **Siapkan foto.** Satu donat per foto, tampak utuh dari atas, dengan cahaya merata.
   Latar apa pun akan dihapus otomatis.
3. **Upload image** atau **Take photo.** Analisis memakan waktu beberapa detik.
4. **Baca hasilnya.** Jenis donat, status lolos atau tidak lolos, persentase, dan tingkat keyakinan.

**Membaca tingkat keyakinan**

- **HIGH:** model cukup yakin, hasil bisa dipakai sebagai acuan.
- **MEDIUM:** cek ulang dengan foto kedua dari posisi berbeda.
- **LOW:** model ragu, periksa donat secara manual.
""")


def page_about():
    st.title("About")
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Apa ini?")
        st.write("Donat Classification membantu inspeksi kualitas donat lewat foto. Aplikasi membaca satu foto, "
                 "lalu menentukan jenis donat dan apakah donat itu lolos atau tidak.")
        st.subheader("Enam hasil yang dikenali")
        st.markdown("- **Donat cincin:** lolos, penyok, atau tidak simetris\n"
                    "- **Donat isi:** lolos, penyok, atau isian bocor")
    with b:
        st.subheader("Cara kerjanya")
        st.markdown("1. Latar belakang dihapus dengan *rembg* dan diganti putih.\n"
                    "2. Foto dijadikan persegi dan diubah ke 224×224 piksel.\n"
                    "3. Nilai warna dinormalisasi sesuai model.\n"
                    "4. Vision Transformer (ViT) hasil fine-tuning memberi peluang untuk tiap kelas.")
        st.subheader("Perlu diingat")
        st.write("Model dilatih dari foto dengan kondisi tertentu. Hasil paling baik untuk foto dengan "
                 "pencahayaan dan sudut serupa. Hasil ini pendamping, bukan pengganti pemeriksaan langsung.")


# ---------------------------------------------------------------- navigasi
st.sidebar.markdown("## Donat Classification")
page = st.sidebar.radio("Menu", ["Classify", "Tutorial", "About"], label_visibility="collapsed")
{"Classify": page_home, "Tutorial": page_tutorial, "About": page_about}[page]()
