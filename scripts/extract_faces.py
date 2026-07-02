"""
Crop + align the face out of every gallery image and save it, so you can inspect
what the recognizer actually sees (full-body shots become tight aligned faces).

Reads  celebrity/<identity>/*
Writes celebrity_faces/<identity>/*   (112x112 aligned faces)

Run:  ./myenv/bin/python extract_faces.py
"""
import os
import cv2
import faceutil

SRC = "celebrity"
DST = "celebrity_faces"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")
SKIP = {"all", "train", "validation", "images", "_clean", "celebrities.csv", "players.csv", ".DS_Store"}

players = sorted(d for d in os.listdir(SRC)
                 if os.path.isdir(os.path.join(SRC, d)) and d not in SKIP)

found = total = 0
missed = []
for player in players:
    src_dir = os.path.join(SRC, player)
    dst_dir = os.path.join(DST, player)
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if not f.lower().endswith(IMG_EXT):
            continue
        total += 1
        img = cv2.imread(os.path.join(src_dir, f))
        if img is None:
            missed.append(f"{player}/{f} (unreadable)")
            continue
        _, _, aligned = faceutil.embed(img)
        if aligned is None:
            missed.append(f"{player}/{f}")
            continue
        found += 1
        cv2.imwrite(os.path.join(dst_dir, os.path.splitext(f)[0] + ".jpg"), aligned)

print(f"Extracted {found}/{total} faces -> {DST}/")
for m in missed:
    print(f"    no face: {m}")
