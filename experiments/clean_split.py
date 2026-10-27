"""
Remove train/val/test leakage found by run_all.py's dedup step.

Every edge in dedup_report.csv (exact MD5 twin, pHash near-duplicate, flipped /
rotated near-duplicate, or Roboflow copies of the same source image) joins two
images into one group. A group that spans several splits is moved entirely into
one split, so no image has a twin on the other side of a split boundary:

    group touches train      → whole group goes to train
    group spans val + test   → whole group goes to val

--mode drop removes the leaked val/test images instead of moving them.
The input dataset is never modified; a cleaned copy is written to --out.

Usage:
    python experiments/clean_split.py --data dataset_v2 --report experiments/out/dedup_report.csv --out dataset_clean
"""

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLITS   = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
CLASSES  = ["sugarcane", "wheat", "maize", "rice", "soybean"]


def find(parent: dict, x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def main():
    ap = argparse.ArgumentParser(description="Move or drop leaked val/test images")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True, help="dedup_report.csv from run_all.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["move", "drop"], default="move")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.out.exists():
        sys.exit(f"{args.out} already exists — remove it or pick another --out")

    parent = {}
    with open(args.report, encoding="utf-8") as f:
        for e in csv.DictReader(f):
            a, b = (e["split_a"], e["file_a"]), (e["split_b"], e["file_b"])
            parent[find(parent, a)] = find(parent, b)

    groups = defaultdict(list)
    for node in list(parent):
        groups[find(parent, node)].append(node)

    dest = {}                         # (split, file) → new split or None (dropped)
    sizes = []
    for members in groups.values():
        splits = {s for s, _ in members}
        if len(splits) < 2:
            continue
        sizes.append(len(members))
        target = "train" if "train" in splits else "val"
        for s, f in members:
            if s != target:
                dest[(s, f)] = target if args.mode == "move" else None

    moves = Counter((s, d) for (s, _), d in dest.items())
    print(f"groups spanning splits: {len(sizes)}  (largest {max(sizes, default=0)} images)")
    for (s, d), n in sorted(moves.items(), key=str):
        print(f"  {s:5s} → {d or 'dropped'}: {n}")

    log = []
    for split in SPLITS:
        for img in sorted((args.data / "images" / split).iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            new = dest.get((split, img.name), split)
            if new is None:
                log.append({"file": img.name, "from": split, "to": "dropped"})
                continue
            if new != split:
                log.append({"file": img.name, "from": split, "to": new})
            for kind, name in (("images", img.name), ("labels", img.stem + ".txt")):
                src = args.data / kind / split / name
                if src.exists():
                    (args.out / kind / new).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, args.out / kind / new / name)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (args.out / "crop_dataset.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\n\nnc: {len(CLASSES)}\nnames:\n{names}\n")

    with open(args.out / "split_changes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "from", "to"])
        w.writeheader()
        w.writerows(log)

    for split in SPLITS:
        d = args.out / "images" / split
        print(f"  {split:5s}: {len(list(d.iterdir())) if d.exists() else 0} images")
    print(f"[✓] cleaned dataset → {args.out}  (changes in split_changes.csv)")


if __name__ == "__main__":
    main()
