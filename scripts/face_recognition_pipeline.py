"""
Build the face look-alike gallery from players/<player>/ and save it.

For each player we store the mean (L2-normalized) SFace embedding of their photos.
A query face (e.g. your selfie) is matched to the nearest player by cosine similarity.

Quality is estimated with leave-one-out: each gallery image is matched while its own
photo is excluded from its player's mean, so we never score against a copy of itself.

Run:  ./myenv/bin/python face_recognition_pipeline.py
"""
import os
import pickle
import numpy as np
import faceutil

ROOT = "celebrity"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")
SKIP = {"all", "train", "validation", "images", "_clean", "celebrities.csv", "players.csv", ".DS_Store"}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


# ---- embed every gallery face --------------------------------------------
print("Embedding gallery faces...")
players = sorted(d for d in os.listdir(ROOT)
                 if os.path.isdir(os.path.join(ROOT, d)) and d not in SKIP)

embs, missed = {}, 0
for p in players:
    pdir = os.path.join(ROOT, p)
    vecs = []
    for f in os.listdir(pdir):
        if not f.lower().endswith(IMG_EXT):
            continue
        e = faceutil.embed_path(os.path.join(pdir, f))
        if e is None:
            missed += 1
        else:
            vecs.append(e)
    embs[p] = vecs

labels = [p for p in players if embs[p]]
centroids = np.stack([unit(np.mean(embs[p], axis=0)) for p in labels])
idx = {p: i for i, p in enumerate(labels)}
print(f"Gallery: {len(labels)} identities, "
      f"{sum(len(embs[p]) for p in labels)} faces ({missed} images had no face)")

# ---- leave-one-out quality estimate --------------------------------------
correct = top5 = total = 0
for p in labels:
    vecs = embs[p]
    if len(vecs) < 2:
        continue  # can't leave one out with a single image
    for i, e in enumerate(vecs):
        loo = unit(np.mean([v for j, v in enumerate(vecs) if j != i], axis=0))
        sims = centroids @ e
        sims[idx[p]] = loo @ e            # score this player without the held-out image
        order = np.argsort(sims)[::-1]
        total += 1
        correct += labels[order[0]] == p
        top5 += p in [labels[k] for k in order[:5]]

print(f"\nLeave-one-out over {total} faces:")
print(f"  Top-1: {correct/total:.3f}   (random = {1/len(labels):.3f})")
print(f"  Top-5: {top5/total:.3f}")

# ---- save the gallery ----------------------------------------------------
with open("face_recognizer.pkl", "wb") as fh:
    pickle.dump({"labels": labels, "centroids": centroids,
                 "yunet": faceutil.YUNET, "sface": faceutil.SFACE}, fh)
print("\nSaved face_recognizer.pkl")
