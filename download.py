from roboflow import Roboflow
from pathlib import Path
import os
import random
import shutil

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("maize-specific").project("maize-leaf-diseases")
dataset = project.version(1).download("yolov8", location="maize_full")

SOURCE = Path("maize_full")
DEST = Path("maize_1200")
TARGET_TOTAL = 1200
SPLITS = ["train", "valid", "test"]

random.seed(42)  # reproducible subset

counts = {
    split: len(list((SOURCE / split / "images").glob("*")))
    for split in SPLITS
    if (SOURCE / split / "images").exists()
}

total = sum(counts.values())
if total < TARGET_TOTAL:
    raise ValueError(f"Dataset only has {total} images, fewer than {TARGET_TOTAL}.")

# Proportional allocation, with leftovers assigned to make exactly 1200.
allocation = {s: int(TARGET_TOTAL * n / total) for s, n in counts.items()}
remaining = TARGET_TOTAL - sum(allocation.values())

for split in sorted(counts, key=lambda s: (TARGET_TOTAL * counts[s] / total) % 1, reverse=True):
    if remaining == 0:
        break
    allocation[split] += 1
    remaining -= 1

for split, n_sample in allocation.items():
    img_dir = SOURCE / split / "images"
    lbl_dir = SOURCE / split / "labels"

    images = [p for p in img_dir.iterdir() if p.is_file()]
    sampled = random.sample(images, n_sample)

    out_img_dir = DEST / split / "images"
    out_lbl_dir = DEST / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sampled:
        shutil.copy2(image_path, out_img_dir / image_path.name)

        label_path = lbl_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, out_lbl_dir / label_path.name)

    print(f"{split}: sampled {len(sampled)} images")

# Keep the dataset configuration/classes.
shutil.copy2(SOURCE / "data.yaml", DEST / "data.yaml")
print(f"Done — exactly {TARGET_TOTAL} images saved to {DEST}/")