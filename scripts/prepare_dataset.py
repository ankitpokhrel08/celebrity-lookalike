"""
Clean players/ into a single deduped gallery folder per player.

For the look-alike app there is no train/validation split -- the gallery is just
every reference image of each player. This script consolidates whatever is under
players/ today (raw folders and/or a previous all|train|validation layout),
removes duplicate images (by content hash), and produces:

  players/<player>/<contenthash>.ext     # one clean folder per player
  celebrities.csv                            # columns: index, image, player

Re-run it any time you add more images (e.g. after scraping footballers into
players/<name>/): it re-consolidates and de-duplicates idempotently.

Run:  ./myenv/bin/python prepare_dataset.py
"""
import os
import csv
import shutil
import hashlib

ROOT = "celebrity"
STAGING = os.path.join(ROOT, "_clean")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")
# folders that are not themselves a "player"
LAYOUT = {"all", "train", "validation", "images", "_clean"}
SKIP = LAYOUT | {"celebrities.csv", ".DS_Store"}


def images_in(d):
    return [f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]


# ---- gather every image per player, from any current layout --------------
sources = {}   # player -> list of file paths


def collect(container):
    """Treat each subdir of `container` as a player folder."""
    for name in os.listdir(container):
        d = os.path.join(container, name)
        if os.path.isdir(d):
            sources.setdefault(name, []).extend(os.path.join(d, f) for f in images_in(d))


for name in os.listdir(ROOT):                      # raw player folders directly under players/
    d = os.path.join(ROOT, name)
    if os.path.isdir(d) and name not in SKIP:
        sources.setdefault(name, []).extend(os.path.join(d, f) for f in images_in(d))
for layout in ("all", "train", "validation"):      # previous split/consolidated layouts
    d = os.path.join(ROOT, layout)
    if os.path.isdir(d):
        collect(d)

# ---- dedupe into staging -------------------------------------------------
if os.path.isdir(STAGING):
    shutil.rmtree(STAGING)

gallery = {}   # player -> list of filenames
for player, paths in sources.items():
    dst = os.path.join(STAGING, player)
    os.makedirs(dst, exist_ok=True)
    seen = set()
    for p in sorted(paths):
        with open(p, "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()[:12]
        if h in seen:
            continue
        seen.add(h)
        ext = os.path.splitext(p)[1].lower()
        ext = ".jpg" if ext == ".jpeg" else ext
        shutil.copy2(p, os.path.join(dst, h + ext))
    gallery[player] = sorted(os.listdir(dst))

removed = sum(len(v) for v in sources.values()) - sum(len(v) for v in gallery.values())
print(f"{len(gallery)} identities, {sum(len(v) for v in gallery.values())} unique images "
      f"({removed} duplicates removed)")

# ---- replace players/ contents with the clean gallery --------------------
for name in os.listdir(ROOT):
    if name in ("_clean", "celebrities.csv", ".DS_Store"):
        continue
    path = os.path.join(ROOT, name)
    shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

rows = []
for player in sorted(gallery):
    os.rename(os.path.join(STAGING, player), os.path.join(ROOT, player))
    for f in gallery[player]:
        rows.append((f, player))
shutil.rmtree(STAGING)

with open(os.path.join(ROOT, "celebrities.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["", "image", "player"])
    for i, (img, player) in enumerate(rows):
        w.writerow([i, img, player])
print(f"Clean gallery written under {ROOT}/<player>/ ; celebrities.csv has {len(rows)} rows")

thin = [p for p, f in gallery.items() if len(f) < 3]
if thin:
    print(f"Players with <3 images (weak gallery): {len(thin)} -> {sorted(thin)}")
