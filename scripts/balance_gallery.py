"""
Clean + balance oversized identity folders (e.g. freshly pasted footballers or actors).

  1. normalize folder names to snake_case
  2. for any folder with more than CAP images, keep up to CAP images that actually
     contain a detectable face (YuNet) and delete the rest -- this both balances the
     dataset (some sets ship 50+ images per identity, others only ~10) and drops
     images with no clear face

Folders already <= CAP are left untouched (don't shrink the thin folders).

Run:  ./myenv/bin/python balance_gallery.py
"""
import os
import re
import random
import cv2
import faceutil

ROOT = "celebrity"
CAP = 20
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")
SKIP = {"all", "train", "validation", "images", "_clean", "celebrities.csv", "players.csv", ".DS_Store"}
random.seed(42)


def snake(name):
    return re.sub(r"[\s\-]+", "_", name.strip().lower()).strip("_")


# ---- 1. normalize folder names -------------------------------------------
for name in list(os.listdir(ROOT)):
    src = os.path.join(ROOT, name)
    if not os.path.isdir(src) or name in SKIP:
        continue
    s = snake(name)
    if s != name:
        dst = os.path.join(ROOT, s)
        if os.path.exists(dst) and not os.path.samefile(src, dst):   # different existing folder -> merge
            for f in os.listdir(src):
                os.rename(os.path.join(src, f), os.path.join(dst, f))
            os.rmdir(src)
        else:                                          # simple or case-only rename (case-insensitive FS)
            tmp = os.path.join(ROOT, s + "__tmp_rename")
            os.rename(src, tmp)
            os.rename(tmp, dst)
        print(f"renamed {name!r} -> {s!r}")

# ---- 2. cap oversized folders to CAP face-detectable images --------------
for name in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d) or name in SKIP:
        continue
    imgs = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]
    if len(imgs) <= CAP:
        continue

    random.shuffle(imgs)
    keep = []
    for f in imgs:
        img = cv2.imread(os.path.join(d, f))
        if img is not None and faceutil.embed(img)[0] is not None:
            keep.append(f)
            if len(keep) >= CAP:
                break
    keep = set(keep)
    deleted = 0
    for f in imgs:
        if f not in keep:
            os.remove(os.path.join(d, f))
            deleted += 1
    print(f"{name:22s} {len(imgs):4d} -> kept {len(keep)} (face-filtered), deleted {deleted}")

print("\nDone. Now run prepare_dataset.py then face_recognition_pipeline.py")
