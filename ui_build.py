"""
Menyiapkan folder antarmuka untuk komponen Streamlit.

static/ (index.html, style.css, app.js buatanmu) disalin APA ADANYA ke folder sementara,
lalu satu tag <script> ditambahkan ke index.html supaya streamlit_bridge.js jalan lebih dulu.
Jadi kalau HTML/CSS/JS diubah, cukup edit static/ saja.
"""
import os
import re
import shutil


def build_ui(static_dir, bridge_path, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(static_dir, out_dir)
    shutil.copy(bridge_path, os.path.join(out_dir, "streamlit_bridge.js"))

    idx = os.path.join(out_dir, "index.html")
    with open(idx, encoding="utf-8") as f:
        html = f.read()
    tag = '<script src="streamlit_bridge.js"></script>\n'
    if "streamlit_bridge.js" not in html:
        if re.search(r"</head>", html, re.I):
            html = re.sub(r"</head>", tag + "</head>", html, count=1, flags=re.I)
        else:
            html = tag + html
    with open(idx, "w", encoding="utf-8") as f:
        f.write(html)
    return out_dir
