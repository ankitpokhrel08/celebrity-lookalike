"""Streamlit look-alike app: upload or capture a face, see who you resemble.

Run locally (from the serve/ directory):
    streamlit run app.py
"""
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from recognizer import Recognizer

st.set_page_config(page_title="Which player do you look like?", layout="centered")


@st.cache_resource
def load_recognizer():
    return Recognizer("assets")


rec = load_recognizer()

st.title("Which player do you look like?")
st.caption(f"Compare your face against {len(rec.labels)} cricketers, footballers, and actors.")
st.info("Your photo is processed only to compute a match and is never stored.")


def to_bgr(file):
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def show_results(img_bgr):
    box, results = rec.match(img_bgr, topk=5)
    disp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 200, 0), 3)

    left, right = st.columns(2)
    left.image(disp, caption="Your photo", use_container_width=True)

    if not results:
        right.warning("No face detected. Try a clearer, front-facing photo.")
        return

    top = results[0]
    right.subheader(f"Closest match: {top['name']}")
    right.metric("Similarity", f"{top['similarity'] * 100:.0f}%")
    if top["similarity"] < 0.30:
        right.caption("No strong match — treat this as a fun guess.")

    st.divider()
    st.subheader("Top 5 look-alikes")
    for col, r in zip(st.columns(len(results)), results):
        col.image(r["thumbnail"], use_container_width=True)
        col.caption(f"{r['name']}  \n{r['similarity'] * 100:.0f}%")


upload_tab, camera_tab = st.tabs(["Upload", "Camera"])
with upload_tab:
    file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp"])
    if file:
        show_results(to_bgr(file))
with camera_tab:
    shot = st.camera_input("Take a photo")
    if shot:
        show_results(to_bgr(shot))
