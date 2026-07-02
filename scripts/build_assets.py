"""Build the deployable asset bundle the app serves from.

Copies the face models and gallery centroids into assets/, and renders one
aligned 112x112 thumbnail per player so the app can show matches without shipping the
whole celebrity/ dataset.

Run (from the project root):  ./myenv/bin/python scripts/build_assets.py
"""
import os
import shutil
import pickle

import cv2
import faceutil

ASSETS = "assets"
THUMBS = os.path.join(ASSETS, "thumbnails")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")

os.makedirs(THUMBS, exist_ok=True)

# 1. models + centroids
shutil.copy2(faceutil.YUNET, os.path.join(ASSETS, os.path.basename(faceutil.YUNET)))
shutil.copy2(faceutil.SFACE, os.path.join(ASSETS, os.path.basename(faceutil.SFACE)))
shutil.copy2("face_recognizer.pkl", os.path.join(ASSETS, "face_recognizer.pkl"))

# 2. one aligned thumbnail per player
labels = pickle.load(open("face_recognizer.pkl", "rb"))["labels"]
made, missing = 0, []
for label in labels:
    d = os.path.join("celebrity", label)
    saved = False
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(IMG_EXT):
            continue
        img = cv2.imread(os.path.join(d, f))
        if img is None:
            continue
        _, _, aligned = faceutil.embed(img)
        if aligned is not None:
            cv2.imwrite(os.path.join(THUMBS, label + ".jpg"), aligned)
            made += 1
            saved = True
            break
    if not saved:
        missing.append(label)

print(f"Assets written to {ASSETS}/")
print(f"  models + face_recognizer.pkl copied")
print(f"  thumbnails: {made}/{len(labels)}")
if missing:
    print(f"  MISSING thumbnails (no detectable face): {missing}")
