"""Self-contained inference for the look-alike service.

Loads the YuNet + SFace ONNX models and the gallery centroids from a local assets
directory (no network, no dependency on the training scripts). Thread-safe: the
OpenCV model objects are not safe to share across threads, so detection/embedding is
guarded by a lock.
"""
import os
import pickle
import threading
import urllib.request

import numpy as np
import cv2

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

# The ONNX weights are large, so they are not committed; fetch them on first boot
# (e.g. on Streamlit Community Cloud) from the OpenCV Zoo.
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_MODEL_URLS = {
    YUNET_FILE: f"{_ZOO}/face_detection_yunet/{YUNET_FILE}",
    SFACE_FILE: f"{_ZOO}/face_recognition_sface/{SFACE_FILE}",
}


def _ensure_models(assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    for fname, url in _MODEL_URLS.items():
        path = os.path.join(assets_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)


class Recognizer:
    def __init__(self, assets_dir="assets"):
        _ensure_models(assets_dir)
        yunet = os.path.join(assets_dir, YUNET_FILE)
        sface = os.path.join(assets_dir, SFACE_FILE)
        with open(os.path.join(assets_dir, "face_recognizer.pkl"), "rb") as f:
            data = pickle.load(f)
        self.labels = list(data["labels"])
        self.centroids = np.asarray(data["centroids"], dtype=np.float32)
        self.thumb_dir = os.path.join(assets_dir, "thumbnails")
        self._detector = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.6, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(sface, "")
        self._lock = threading.Lock()

    def _detect_best(self, img):
        """Largest face with a recall fallback: ease the threshold, then 2x upscale."""
        def largest(faces):
            return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]

        for score in (0.6, 0.3):
            h, w = img.shape[:2]
            self._detector.setInputSize((w, h))
            self._detector.setScoreThreshold(score)
            _, faces = self._detector.detect(img)
            if faces is not None and len(faces):
                return largest(faces), img, 1.0

        big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        h, w = big.shape[:2]
        self._detector.setInputSize((w, h))
        self._detector.setScoreThreshold(0.5)
        _, faces = self._detector.detect(big)
        if faces is not None and len(faces):
            return largest(faces), big, 2.0
        return None, None, None

    def match(self, img_bgr, topk=5):
        """Match one face image (HxWx3 BGR).

        Returns (box_xywh_or_None, results) where results is a list of dicts:
        {label, name, similarity, thumbnail}, ranked most similar first. Empty if no
        face was detected.
        """
        with self._lock:
            face, used, scale = self._detect_best(img_bgr)
            if face is None:
                return None, []
            aligned = self._recognizer.alignCrop(used, face)
            vec = self._recognizer.feature(aligned).flatten()

        vec = vec / (np.linalg.norm(vec) + 1e-9)
        sims = self.centroids @ vec
        order = np.argsort(sims)[::-1][:topk]
        box = (face[:4] / scale).astype(int)
        results = [{
            "label": self.labels[i],
            "name": self.labels[i].replace("_", " ").title(),
            "similarity": float(sims[i]),
            "thumbnail": os.path.join(self.thumb_dir, self.labels[i] + ".jpg"),
        } for i in order]
        return box, results
