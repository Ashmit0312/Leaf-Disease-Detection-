"""
Merge 5 separate crop YOLO datasets into one unified dataset.

Handles Roboflow's ACTUAL export structure (split-first, not images-first):
    Sugar-Cane-Disease-Model-2/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
        data.yaml

Also supports the alternative images-first layout:
    sugarcane_dataset/
        images/train/, images/val/, images/test/
        labels/train/, labels/val/, labels/test/

Output (always images-first, matching train.py):
    dataset/
        images/train/, images/val/, images/test/
        labels/train/, labels/val/, labels/test/
        crop_dataset.yaml
"""

import os, shutil, random
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

# Map each crop folder → class ID
# Change these paths to wherever your 5 downloaded Roboflow folders actually are
CROP_SOURCES = {
    0: "Sugar-Cane-Disease-Model-2",
    1: "wheat_dataset",
    2: "maize_dataset",
    3: "rice_dataset",
    4: "soybean_dataset",
}

OUTPUT_DIR = "dataset"
SPLITS     = ["train", "val", "test"]
CLASS_NAMES = ["sugarcane", "wheat", "maize", "rice", "soybean"]

# Only needed if your source folders have NO train/val/test split
# (i.e. all images are in a flat folder)
FLAT_SPLIT_RATIO = (0.70, 0.20, 0.10)   # train / val / test


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_dirs():
    for split in SPLITS:
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)
    print(f"[✓] Output directories created under '{OUTPUT_DIR}/'")


def detect_layout(src: str) -> str:
    """
    Detect which folder layout this Roboflow/YOLO dataset uses.
    Returns: "split_first" | "images_first" | "flat" | "unknown"
    """
    # Roboflow's actual export layout: train/images/, valid/images/, test/images/
    if Path(src, "train", "images").exists():
        return "split_first"

    # Alternative layout: images/train/, labels/train/
    if Path(src, "images", "train").exists():
        return "images_first"

    # Flat: images sitting directly in the folder or in images/
    if Path(src, "images").exists() or any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in Path(src).glob("*")
    ):
        return "flat"

    return "unknown"


def copy_split_first_dataset(class_id: int, src: str):
    """Copy from Roboflow's actual layout: train/images/, valid/images/, test/images/"""
    crop_name = CLASS_NAMES[class_id]
    count = 0

    # Roboflow uses "valid" not "val" — map it here
    roboflow_split_map = {"train": "train", "val": "valid", "test": "test"}

    for our_split, rf_split in roboflow_split_map.items():
        img_dir = Path(src, rf_split, "images")
        lbl_dir = Path(src, rf_split, "labels")

        if not img_dir.exists():
            print(f"  [!] Missing {img_dir}, skipping")
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            stem    = img_path.stem
            lbl_src = lbl_dir / f"{stem}.txt"

            # Unique filename: prepend crop name to avoid collisions across datasets
            new_stem = f"{crop_name}_{stem}"
            img_dst  = Path(OUTPUT_DIR, "images", our_split, new_stem + img_path.suffix)
            lbl_dst  = Path(OUTPUT_DIR, "labels", our_split, new_stem + ".txt")

            shutil.copy2(img_path, img_dst)

            if lbl_src.exists():
                rewrite_label(lbl_src, lbl_dst, class_id)
            else:
                lbl_dst.touch()   # background image, no objects

            count += 1

    print(f"  [✓] class {class_id} ({crop_name} ← {src}): {count} images copied")


def is_split_dataset(src: str) -> bool:
    """Check if source has images/train/ style subfolders (legacy check)."""
    return Path(src, "images", "train").exists()


def copy_images_first_dataset(class_id: int, src: str):
    """Copy from a dataset using images/train/, labels/train/ layout."""
    count = 0
    for split in SPLITS:
        img_dir = Path(src, "images", split)
        lbl_dir = Path(src, "labels", split)

        if not img_dir.exists():
            # try 'valid' instead of 'val' (Roboflow naming)
            if split == "val":
                img_dir = Path(src, "images", "valid")
                lbl_dir = Path(src, "labels", "valid")
            if not img_dir.exists():
                print(f"  [!] Missing {img_dir}, skipping")
                continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            stem    = img_path.stem
            lbl_src = lbl_dir / f"{stem}.txt"

            # Unique filename: prepend class name to avoid collisions
            crop_name = CLASS_NAMES[class_id]
            new_stem  = f"{crop_name}_{stem}"
            img_dst   = Path(OUTPUT_DIR, "images", split, new_stem + img_path.suffix)
            lbl_dst   = Path(OUTPUT_DIR, "labels", split, new_stem + ".txt")

            shutil.copy2(img_path, img_dst)

            if lbl_src.exists():
                # Re-write labels with correct class_id
                rewrite_label(lbl_src, lbl_dst, class_id)
            else:
                # No label file → create empty (background image)
                lbl_dst.touch()

            count += 1

    print(f"  [✓] class {class_id} ({src}): {count} images copied")


def copy_flat_dataset(class_id: int, src: str):
    """
    Source has no split — images sit in a flat folder or images/ directly.
    We'll split randomly.
    """
    crop_name = CLASS_NAMES[class_id]

    # Try common flat layouts
    for candidate in [
        Path(src, "images"),
        Path(src),
    ]:
        imgs = [p for p in candidate.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if imgs:
            img_dir = candidate
            lbl_dir = Path(src, "labels") if Path(src, "labels").exists() else candidate
            break
    else:
        print(f"  [!] No images found in {src}, skipping")
        return

    random.shuffle(imgs)
    n     = len(imgs)
    n_tr  = int(n * FLAT_SPLIT_RATIO[0])
    n_val = int(n * FLAT_SPLIT_RATIO[1])
    split_map = (
        [(p, "train") for p in imgs[:n_tr]] +
        [(p, "val")   for p in imgs[n_tr:n_tr+n_val]] +
        [(p, "test")  for p in imgs[n_tr+n_val:]]
    )

    count = 0
    for img_path, split in split_map:
        stem    = img_path.stem
        lbl_src = lbl_dir / f"{stem}.txt"
        new_stem = f"{crop_name}_{stem}"
        img_dst  = Path(OUTPUT_DIR, "images", split, new_stem + img_path.suffix)
        lbl_dst  = Path(OUTPUT_DIR, "labels", split, new_stem + ".txt")

        shutil.copy2(img_path, img_dst)
        if lbl_src.exists():
            rewrite_label(lbl_src, lbl_dst, class_id)
        else:
            lbl_dst.touch()
        count += 1

    print(f"  [✓] class {class_id} ({src}): {count} images split & copied")


def rewrite_label(src: Path, dst: Path, expected_class_id: int):
    """
    Copy label file, replacing the class ID on every line with expected_class_id.
    Handles single-class datasets where source labels might use class 0.
    """
    lines = src.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[0] = str(expected_class_id)   # override class ID
            new_lines.append(" ".join(parts))
    dst.write_text("\n".join(new_lines))


def write_yaml():
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names:
{names_block}
"""
    out = Path(OUTPUT_DIR, "crop_dataset.yaml")
    out.write_text(yaml_content)
    print(f"[✓] YAML written → {out}")


def print_summary():
    print("\n── Dataset Summary ──────────────────────────────")
    total = 0
    for split in SPLITS:
        n = len(list(Path(OUTPUT_DIR, "images", split).iterdir()))
        print(f"  {split:6s}: {n} images")
        total += n
    print(f"  {'total':6s}: {total} images")
    print("─────────────────────────────────────────────────\n")
    print("Next step:")
    print("  python train.py --mode train --size n --epochs 100 --device 0\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_dirs()
    print("\n[→] Merging datasets...\n")

    for class_id, src in CROP_SOURCES.items():
        if not Path(src).exists():
            print(f"  [!] Folder not found: '{src}' — skipping class {class_id}")
            continue

        layout = detect_layout(src)
        print(f"  Detected layout for '{src}': {layout}")

        if layout == "split_first":
            copy_split_first_dataset(class_id, src)
        elif layout == "images_first":
            copy_images_first_dataset(class_id, src)
        elif layout == "flat":
            copy_flat_dataset(class_id, src)
        else:
            print(f"  [!] Could not detect folder structure for '{src}' — skipping. "
                  f"Check that it contains train/images/, images/train/, or loose image files.")

    write_yaml()
    print_summary()
