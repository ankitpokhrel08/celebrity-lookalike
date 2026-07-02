"""Streamlit front end for the facial similarity search.

Run locally (from the project root):
    streamlit run app.py
"""
import base64
import time

import numpy as np
import cv2
import streamlit as st
from PIL import Image

from recognizer import Recognizer

SCAN_SECONDS = 2.0

st.set_page_config(page_title="Facial Similarity Search", layout="centered")

# Minimal chrome: hide the Streamlit menu, footer, and toolbar.
st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 46rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_recognizer():
    return Recognizer("assets")


rec = load_recognizer()

st.title("Facial Similarity Search")
st.markdown(
    f"Upload a photograph to find the closest matches among **{len(rec.labels)}** "
    "cricketers, footballers, and actors. Faces are compared by cosine similarity "
    "of 128-dimensional embeddings; no photo is ever stored."
)


def to_bgr(file):
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def scan_html(img_bgr):
    """Photo with an animated scan line, rendered as inline HTML."""
    h, w = img_bgr.shape[:2]
    if w > 480:  # keep the base64 payload small
        img_bgr = cv2.resize(img_bgr, (480, int(h * 480 / w)))
    _, buf = cv2.imencode(".jpg", img_bgr)
    b64 = base64.b64encode(buf).decode()
    return f"""
    <style>
    .scan-wrap {{position: relative; max-width: 320px; margin: 0 auto;
                 overflow: hidden; border-radius: 6px;}}
    .scan-wrap img {{width: 100%; display: block;}}
    .scan-line {{position: absolute; left: 0; width: 100%; height: 2px;
                 background: #1a3d5c;
                 box-shadow: 0 0 14px 5px rgba(26, 61, 92, 0.35);
                 animation: scan {SCAN_SECONDS / 2:.1f}s linear infinite alternate;}}
    @keyframes scan {{from {{top: 0;}} to {{top: calc(100% - 2px);}}}}
    .scan-caption {{text-align: center; color: #6b7280;
                    font-size: 0.85rem; margin-top: 0.6rem;}}
    </style>
    <div class="scan-wrap">
      <img src="data:image/jpeg;base64,{b64}" />
      <div class="scan-line"></div>
    </div>
    <div class="scan-caption">Analyzing photograph</div>
    """


def show_results(img_bgr):
    # Play the scan animation only for a new photo, not on every widget rerun;
    # the match itself is computed while the animation runs.
    token = hash(img_bgr.tobytes())
    animate = st.session_state.get("scanned") != token
    placeholder = st.empty()
    if animate:
        placeholder.markdown(scan_html(img_bgr), unsafe_allow_html=True)
        start = time.time()

    box, results = rec.match(img_bgr, topk=5)

    if animate:
        time.sleep(max(0.0, SCAN_SECONDS - (time.time() - start)))
        placeholder.empty()
        st.session_state["scanned"] = token

    disp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 255, 255), 2)

    photo, summary = st.columns(2, gap="large")
    photo.image(disp, caption="Input photograph", use_container_width=True)

    if not results:
        summary.error("No face detected. Use a clearer, front-facing photograph.")
        return

    top = results[0]
    summary.markdown("**Closest match**")
    summary.markdown(f"### {top['name']}")
    summary.metric("Similarity", f"{top['similarity'] * 100:.1f}%")
    if top["similarity"] < 0.30:
        summary.caption("Low confidence: no identity in the gallery is a strong match.")

    st.divider()
    st.markdown("**Ranked matches**")
    for col, (rank, r) in zip(st.columns(len(results)), enumerate(results, 1)):
        col.image(r["thumbnail"], use_container_width=True)
        col.markdown(f"{rank}. {r['name']}")
        col.progress(min(max(r["similarity"], 0.0), 1.0),
                     text=f"{r['similarity'] * 100:.0f}%")


upload_tab, camera_tab = st.tabs(["Upload photo", "Use camera"])
with upload_tab:
    file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png", "webp"])
    if file:
        show_results(to_bgr(file))
with camera_tab:
    shot = st.camera_input("Capture an image")
    if shot:
        show_results(to_bgr(shot))

st.divider()
st.caption(
    "Detection: YuNet. Embedding: SFace (OpenCV). Matching: cosine similarity against "
    "per-identity centroids. Photographs are processed in memory only."
)
