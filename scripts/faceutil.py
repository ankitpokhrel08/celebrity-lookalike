"""
Shared face detection + embedding, used by BOTH training (gallery) and inference
so the two never drift.

  YuNet  -> find + locate the face (works even in full-body shots)
  SFace  -> align to the 5 landmarks and produce a 128-D face embedding

detect_best() adds a recall fallback for hard images (small / full-body faces):
try the native image at a normal then low score threshold, then a 2x upscale.
"""
import os
import urllib.request
import numpy as np
import cv2

YUNET = "face_models/face_detection_yunet_2023mar.onnx"
SFACE = "face_models/face_recognition_sface_2021dec.onnx"

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_URLS = {
    YUNET: f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    SFACE: f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
os.makedirs("face_models", exist_ok=True)
for _p, _u in _URLS.items():
    if not os.path.exists(_p):
        print(f"Downloading {os.path.basename(_p)} ...")
        urllib.request.urlretrieve(_u, _p)

_detector = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.6, 0.3, 5000)
_recognizer = cv2.FaceRecognizerSF.create(SFACE, "")


def _detect(img, score):
    h, w = img.shape[:2]
    _detector.setInputSize((w, h))
    _detector.setScoreThreshold(score)
    _, faces = _detector.detect(img)
    return faces


def detect_best(img):
    """Return (face_row, image_used, scale) for the largest face, or (None, None, None).

    `face_row` is in the coordinates of `image_used`; `scale` maps those coords
    back to the original image (coord_original = coord_used / scale).
    """
    for score in (0.6, 0.3):                      # native size, easing the threshold
        faces = _detect(img, score)
        if faces is not None and len(faces):
            return faces[np.argmax(faces[:, 2] * faces[:, 3])], img, 1.0

    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # tiny faces
    faces = _detect(big, 0.5)
    if faces is not None and len(faces):
        return faces[np.argmax(faces[:, 2] * faces[:, 3])], big, 2.0
    return None, None, None


def embed(img):
    """Return (unit_embedding, box_xywh_in_original, aligned_face_bgr) or (None, None, None)."""
    face, used, scale = detect_best(img)
    if face is None:
        return None, None, None
    aligned = _recognizer.alignCrop(used, face)
    v = _recognizer.feature(aligned).flatten()
    box = (face[:4] / scale).astype(int)
    return v / (np.linalg.norm(v) + 1e-9), box, aligned


def embed_path(img_path):
    """Convenience: read an image file and return its embedding (or None)."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    return embed(img)[0]
