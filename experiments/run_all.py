"""
Experiments for the paper revision (reviewer response).

Every step writes to --out (default experiments/out/), prints a summary and
keeps going if another step fails. Steps that need weights or an API key are
skipped with a note when those are missing.

    1. dedup     MD5 + pHash leakage audit across splits (no weights needed)
    2. stats     images / boxes / box sizes per crop per split, source table
    3. detect    val + test metrics, per-crop AP, confusion matrix, worst cases
    4. vlm       zero-shot VLM accuracy, YOLO image-level accuracy, agreement
    5. fusion    "YOLO if top conf >= tau else VLM", tau picked on val
    6. latency   ONNX Runtime CPU latency / FPS / peak RSS, optional INT8
    7. plantdoc  cross-dataset check on PlantDoc corn + soybean (optional)
    8. ablation  YOLOv8n vs YOLOv8s (only with --weights-s)

Usage:
    python experiments/run_all.py --data dataset --weights best.pt
    python experiments/run_all.py --data dataset --steps dedup,stats
    python experiments/run_all.py --data dataset --weights best.pt --steps vlm,fusion --vlm-splits test,val

The VLM step reads GROQ_API_KEY from the environment and caches every answer
in out/vlm_cache.jsonl, so reruns resume where the daily quota stopped them.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES   = ["sugarcane", "wheat", "maize", "rice", "soybean"]
SPLITS    = ["train", "val", "test"]
IMG_EXTS  = {".jpg", ".jpeg", ".png"}
TAUS      = [0.25, 0.4, 0.5, 0.6, 0.75]
NEAR_DUP  = 4          # pHash Hamming distance threshold (64-bit hash)
YOLO_CONF = 0.25       # Ultralytics predict default, used for image-level labels
VAL_CONF  = 0.001      # Ultralytics val defaults, stated in the paper
VAL_IOU   = 0.7

# Source exports used by merge_datasets.py, in class-ID order
SOURCES = {
    "sugarcane": "Sugar-Cane-Disease-Model-2",
    "wheat":     "Wheat-plant-diseases-6",
    "maize":     "maize_1200",
    "rice":      "Rice-plant-disease-detection-1",
    "soybean":   "soy-leaf-disease-15",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def crop_of(name: str) -> str:
    """Ground-truth crop from the merged filename prefix (e.g. rice_IMG_...)."""
    return name.split("_", 1)[0]


def source_stem(name: str) -> str:
    """Roboflow augmented copies are <orig>_jpg.rf.<hash>.jpg — group by <orig>."""
    stem = Path(name).stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def list_images(data: Path, split: str) -> list[Path]:
    d = data / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.exists() else []


def read_labels(data: Path, split: str, img: Path) -> list[list[float]]:
    lbl = data / "labels" / split / (img.stem + ".txt")
    rows = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts])
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def dataset_yaml(data: Path, out: Path) -> Path:
    """Write a yaml pointing at --data so results never depend on a stale path."""
    y = out / "eval_dataset.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    y.write_text(f"path: {data.resolve().as_posix()}\ntrain: images/train\n"
                 f"val: images/val\ntest: images/test\n\nnc: {len(CLASSES)}\nnames:\n{names}\n")
    return y


class Results:
    """Collects every reported number with the file it came from."""

    def __init__(self, out: Path):
        self.out = out
        self.path = out / "numbers.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def add(self, step: str, key: str, value, source: str):
        self.data.setdefault(step, {})[key] = {"value": value, "source": source}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))


# ─── 1. Dedup / leakage audit ─────────────────────────────────────────────────

def phash_bits(img) -> int:
    import imagehash
    return int(str(imagehash.phash(img)), 16)


def dihedral(img):
    """The 8 flips/90° rotations — Roboflow augmentations used by our sources."""
    from PIL import Image
    T = Image.Transpose
    rots = [img, img.transpose(T.ROTATE_90), img.transpose(T.ROTATE_180), img.transpose(T.ROTATE_270)]
    return rots + [r.transpose(T.FLIP_LEFT_RIGHT) for r in rots]


def step_dedup(args, res: Results):
    from PIL import Image

    out = args.out
    files = []                               # (split, path)
    for s in SPLITS:
        files += [(s, p) for p in list_images(args.data, s)]
    print(f"  hashing {len(files)} images ...")

    md5s, hashes, dihedral_hashes = [], [], {}
    for i, (s, p) in enumerate(files):
        md5s.append(hashlib.md5(p.read_bytes()).hexdigest())
        with Image.open(p) as im:
            im = im.convert("RGB")
            hashes.append(phash_bits(im))
            if s != "train":                  # query side: all 8 orientations
                dihedral_hashes[i] = [phash_bits(t) for t in dihedral(im)]
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(files)}")

    names  = [p.name for _, p in files]
    splits = [s for s, _ in files]
    H = np.array(hashes, dtype=np.uint64)

    # Released manifest so others can check our split for overlap with theirs
    write_csv(out / "manifest.csv", [{"split": s, "file": n, "crop": crop_of(n), "md5": m, "phash": f"{h:016x}"}
                                     for s, n, m, h in zip(splits, names, md5s, hashes)])

    edges = []                                # dicts written to dedup_report.csv

    def edge(kind, i, j, dist):
        a, b = sorted((i, j))
        edges.append({"type": kind, "split_a": splits[a], "file_a": names[a],
                      "split_b": splits[b], "file_b": names[b], "hamming": dist,
                      "crop_a": crop_of(names[a]), "crop_b": crop_of(names[b])})

    # Exact duplicates (byte-identical)
    by_md5 = defaultdict(list)
    for i, h in enumerate(md5s):
        by_md5[h].append(i)
    for idx in by_md5.values():
        for k in range(1, len(idx)):
            edge("exact_md5", idx[0], idx[k], 0)

    # Near duplicates, same orientation: all pairs, vectorised popcount
    seen = set()
    for i in range(len(H)):
        d = np.bitwise_count(H[i + 1:] ^ H[i])
        for off in np.nonzero(d <= NEAR_DUP)[0]:
            j = i + 1 + int(off)
            if md5s[i] != md5s[j]:
                edge("near_phash", i, j, int(d[off]))
                seen.add((i, j))

    # Near duplicates under flip/rotation: val/test image vs every other image
    for i, variants in dihedral_hashes.items():
        V = np.array(variants[1:], dtype=np.uint64)    # skip identity (done above)
        d = np.bitwise_count(H[None, :] ^ V[:, None]).min(axis=0)
        d[i] = 255
        for j in np.nonzero(d <= NEAR_DUP)[0]:
            j = int(j)
            pair = tuple(sorted((i, j)))
            if pair not in seen and md5s[i] != md5s[j]:
                edge("near_phash_dihedral", i, j, int(d[j]))
                seen.add(pair)

    # Roboflow augmentation groups (same <orig> stem) spanning several splits
    groups = defaultdict(list)
    for i, n in enumerate(names):
        groups[source_stem(n)].append(i)
    multi_split_groups = 0
    for idx in groups.values():
        if len({splits[i] for i in idx}) > 1:
            multi_split_groups += 1
            for k in range(1, len(idx)):
                if splits[idx[k]] != splits[idx[0]]:
                    edge("same_source_stem", idx[0], idx[k], -1)

    write_csv(out / "dedup_report.csv", edges)

    # Summary: which val/test images have a twin in an earlier split
    def pair(e):
        return tuple(sorted((e["split_a"], e["split_b"])))

    counts = Counter((e["type"], pair(e)) for e in edges)
    rows = [{"type": t, "splits": f"{a}-{b}", "pairs": n} for (t, (a, b)), n in sorted(counts.items())]
    write_csv(out / "dedup_summary.csv", rows)

    n_split = Counter(splits)
    leaked = {"val": set(), "test": set()}
    for e in edges:
        a, b = e["split_a"], e["split_b"]
        if a == b:
            continue
        for s, f, other in ((a, e["file_a"], b), (b, e["file_b"], a)):
            # val is leaked if it has a train twin; test if it has a train or val twin
            if (s == "val" and other == "train") or (s == "test" and other in ("train", "val")):
                leaked[s].add(f)
    cross_crop = sum(1 for e in edges if e["crop_a"] != e["crop_b"])

    src = "dedup_report.csv"
    res.add("dedup", "images_total", len(files), src)
    for s in SPLITS:
        res.add("dedup", f"images_{s}", n_split[s], src)
    for s in ("val", "test"):
        res.add("dedup", f"leaked_{s}", len(leaked[s]), src)
        res.add("dedup", f"leaked_{s}_pct", round(100 * len(leaked[s]) / max(n_split[s], 1), 2), src)
    for (t, (a, b)), n in counts.items():
        res.add("dedup", f"pairs_{t}_{a}-{b}", n, "dedup_summary.csv")
    res.add("dedup", "source_groups_spanning_splits", multi_split_groups, src)
    res.add("dedup", "cross_crop_duplicate_pairs", cross_crop, src)
    write_csv(out / "leaked_files.csv",
              [{"split": s, "file": f} for s in ("val", "test") for f in sorted(leaked[s])])

    print("\n  type                      splits       pairs")
    for r in rows:
        print(f"  {r['type']:24s}  {r['splits']:11s}  {r['pairs']}")
    for s in ("val", "test"):
        print(f"  leaked {s}: {len(leaked[s])}/{n_split[s]} "
              f"({100 * len(leaked[s]) / max(n_split[s], 1):.1f}%)")
    print(f"  Roboflow source groups spanning >1 split: {multi_split_groups}")
    print(f"  duplicate pairs across different crops   : {cross_crop}")


# ─── 2. Per-crop statistics ───────────────────────────────────────────────────

def step_stats(args, res: Results):
    import yaml

    rows = []
    for s in SPLITS:
        per = defaultdict(lambda: {"images": 0, "boxes": 0, "empty": 0, "areas": []})
        for p in list_images(args.data, s):
            c = crop_of(p.name)
            labels = read_labels(args.data, s, p)
            per[c]["images"] += 1
            per[c]["boxes"] += len(labels)
            per[c]["empty"] += not labels
            per[c]["areas"] += [r[3] * r[4] for r in labels if len(r) == 5]
        for c in CLASSES:
            a = np.array(per[c]["areas"]) if per[c]["areas"] else np.array([np.nan])
            rows.append({"split": s, "crop": c, "images": per[c]["images"], "boxes": per[c]["boxes"],
                         "background_images": per[c]["empty"],
                         "boxes_per_image": round(per[c]["boxes"] / max(per[c]["images"], 1), 2),
                         "median_box_area_pct": round(100 * float(np.nanmedian(a)), 2),
                         "boxes_lt_1pct_area": int(np.sum(a < 0.01))})
    write_csv(args.out / "class_stats.csv", rows)

    # Per-crop totals over all splits — the dataset table in the paper
    tot = []
    for c in CLASSES:
        r = [x for x in rows if x["crop"] == c]
        tot.append({"crop": c, **{f"{x['split']}_images": x["images"] for x in r},
                    "images": sum(x["images"] for x in r), "boxes": sum(x["boxes"] for x in r),
                    "median_box_area_pct": next(x["median_box_area_pct"] for x in r if x["split"] == "train")})
    write_csv(args.out / "class_totals.csv", tot)

    # Source provenance from each Roboflow export's data.yaml
    src_rows = []
    for c, folder in SOURCES.items():
        y = args.sources / folder / "data.yaml"
        meta = yaml.safe_load(y.read_text()).get("roboflow", {}) if y.exists() else {}
        src_rows.append({"crop": c, "folder": folder, "workspace": meta.get("workspace", ""),
                         "project": meta.get("project", ""), "version": meta.get("version", ""),
                         "license": meta.get("license", ""), "url": meta.get("url", "")})
    write_csv(args.out / "sources.csv", src_rows)

    print(f"\n  {'crop':10s} {'train':>6s} {'val':>5s} {'test':>5s} {'images':>7s} {'boxes':>7s} {'med.area%':>9s}")
    for t in tot:
        print(f"  {t['crop']:10s} {t['train_images']:6d} {t['val_images']:5d} {t['test_images']:5d} "
              f"{t['images']:7d} {t['boxes']:7d} {t['median_box_area_pct']:9.2f}")
        for k in ("images", "boxes", "median_box_area_pct"):
            res.add("stats", f"{t['crop']}_{k}", t[k], "class_totals.csv")
    for s in SPLITS:
        r = [x for x in rows if x["split"] == s]
        res.add("stats", f"{s}_images", sum(x["images"] for x in r), "class_stats.csv")
        res.add("stats", f"{s}_boxes", sum(x["boxes"] for x in r), "class_stats.csv")
        res.add("stats", f"{s}_background", sum(x["background_images"] for x in r), "class_stats.csv")
    for r in src_rows:
        print(f"  {r['crop']:10s} {r['license']:10s} {r['url']}")


# ─── 3. Detector evaluation ───────────────────────────────────────────────────

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def predict_split(args, split: str) -> dict:
    """Run the detector once per image and cache boxes in yolo_preds_<split>.json."""
    cache = args.out / f"yolo_preds_{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO
    model = YOLO(args.weights)
    preds = {}
    for p in list_images(args.data, split):
        r = model.predict(str(p), conf=0.05, iou=VAL_IOU, imgsz=640, verbose=False, device="cpu")[0]
        preds[p.name] = {"xyxyn": r.boxes.xyxyn.tolist(), "cls": r.boxes.cls.int().tolist(),
                         "conf": [round(c, 4) for c in r.boxes.conf.tolist()]}
    cache.write_text(json.dumps(preds))
    return preds


def image_label(pred: dict, conf: float = YOLO_CONF) -> tuple[str, float]:
    """Image-level crop = class of the top-confidence box, or 'none'."""
    if not pred["conf"] or max(pred["conf"]) < conf:
        return "none", max(pred["conf"], default=0.0)
    k = int(np.argmax(pred["conf"]))
    return CLASSES[pred["cls"][k]], pred["conf"][k]


def step_detect(args, res: Results):
    from ultralytics import YOLO

    yaml_path = dataset_yaml(args.data, args.out)
    rows = []
    for split in ("val", "test"):
        m = YOLO(args.weights).val(data=str(yaml_path), split=split, imgsz=640, batch=16,
                                   conf=VAL_CONF, iou=VAL_IOU, plots=True, device=args.device,
                                   project=str(args.out / "yolo_val"), name=split, exist_ok=True)
        b = m.box
        rows.append({"split": split, "class": "all", "P": round(b.mp, 4), "R": round(b.mr, 4),
                     "mAP50": round(b.map50, 4), "mAP50_95": round(b.map, 4)})
        for i, c in enumerate(m.ap_class_index):
            p, r, ap50, ap = b.class_result(i)
            rows.append({"split": split, "class": CLASSES[int(c)], "P": round(p, 4), "R": round(r, 4),
                         "mAP50": round(ap50, 4), "mAP50_95": round(ap, 4)})
        save_dir = Path(m.save_dir)
        for pat in ("confusion_matrix_normalized.png", "*PR_curve.png"):
            for f in save_dir.glob(pat):
                shutil.copy2(f, args.out / f"{split}_{f.name}")
        # Speed per image (ms) as measured by Ultralytics on this device
        res.add("detect", f"{split}_speed_ms", {k: round(v, 2) for k, v in m.speed.items()}, "det_metrics.csv")

    write_csv(args.out / "det_metrics.csv", rows)
    print(f"\n  conf={VAL_CONF}  NMS IoU={VAL_IOU}")
    print(f"  {'split':5s} {'class':10s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>8s}")
    for r in rows:
        print(f"  {r['split']:5s} {r['class']:10s} {r['P']:6.3f} {r['R']:6.3f} {r['mAP50']:6.3f} {r['mAP50_95']:8.3f}")
        for k in ("P", "R", "mAP50", "mAP50_95"):
            res.add("detect", f"{r['split']}_{r['class']}_{k}", r[k], "det_metrics.csv")
    res.add("detect", "val_conf", VAL_CONF, "run_all.py")
    res.add("detect", "val_nms_iou", VAL_IOU, "run_all.py")

    worst_cases(args, res)


def worst_cases(args, res: Results, n: int = 10):
    """Per-image matching at IoU 0.5 / conf 0.25; save the n worst test images."""
    import cv2

    preds = predict_split(args, "test")
    rows = []
    for p in list_images(args.data, "test"):
        gt = read_labels(args.data, "test", p)
        g_cls = np.array([int(r[0]) for r in gt if len(r) == 5])
        g_box = np.array([[r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2]
                          for r in gt if len(r) == 5]).reshape(-1, 4)
        pr = preds[p.name]
        keep = [k for k, c in enumerate(pr["conf"]) if c >= YOLO_CONF]
        p_box = np.array([pr["xyxyn"][k] for k in keep]).reshape(-1, 4)
        p_cls = np.array([pr["cls"][k] for k in keep])

        iou = box_iou(p_box, g_box)
        tp = wrong_cls = 0
        used = set()
        for k in np.argsort([-pr["conf"][j] for j in keep]):
            if not len(g_box):
                break
            j = int(np.argmax(iou[k]))
            if iou[k, j] >= 0.5 and j not in used:
                used.add(j)
                if p_cls[k] == g_cls[j]:
                    tp += 1
                else:
                    wrong_cls += 1
        fp, fn = len(keep) - tp, len(g_box) - tp
        rows.append({"file": p.name, "crop": crop_of(p.name), "gt_boxes": len(g_box), "pred_boxes": len(keep),
                     "tp": tp, "fp": fp, "fn": fn, "wrong_crop_matches": wrong_cls,
                     "pred_crops": "|".join(sorted({CLASSES[c] for c in p_cls})),
                     "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3)})

    write_csv(args.out / "test_per_image.csv", rows)
    worst = sorted(rows, key=lambda r: (r["f1"], -(r["fp"] + r["fn"])))[:n]
    write_csv(args.out / "worst_cases.csv", worst)

    wdir = args.out / "worst_cases"
    wdir.mkdir(exist_ok=True)
    for r in worst:
        img = cv2.imread(str(args.data / "images" / "test" / r["file"]))
        h, w = img.shape[:2]
        for g in read_labels(args.data, "test", Path(r["file"])):
            x0, y0, x1, y1 = (g[1] - g[3] / 2) * w, (g[2] - g[4] / 2) * h, (g[1] + g[3] / 2) * w, (g[2] + g[4] / 2) * h
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), 2)
        pr = preds[r["file"]]
        for b, c, s in zip(pr["xyxyn"], pr["cls"], pr["conf"]):
            if s >= YOLO_CONF:
                cv2.rectangle(img, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), (0, 0, 230), 2)
                cv2.putText(img, f"{CLASSES[c]} {s:.2f}", (int(b[0] * w), max(int(b[1] * h) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1)
        cv2.imwrite(str(wdir / r["file"]), img)

    # Aggregate error types for the error-analysis paragraph
    tot = {k: sum(r[k] for r in rows) for k in ("gt_boxes", "tp", "fp", "fn", "wrong_crop_matches")}
    images_wrong_crop = sum(1 for r in rows if r["pred_crops"] and r["pred_crops"] != r["crop"])
    for k, v in tot.items():
        res.add("errors", f"test_{k}", v, "test_per_image.csv")
    res.add("errors", "test_images_with_other_crop_pred", images_wrong_crop, "test_per_image.csv")
    res.add("errors", "test_images_no_pred", sum(1 for r in rows if r["pred_boxes"] == 0), "test_per_image.csv")
    print(f"\n  test @conf {YOLO_CONF}, IoU 0.5: {tot}; images with another crop predicted: {images_wrong_crop}")
    print(f"  worst {n} cases → {wdir}")


# ─── 4. VLM evaluation ────────────────────────────────────────────────────────

def load_vlm_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[(r["split"], r["file"])] = r
    return cache


def run_vlm(args, split: str, cache: dict, budget: list) -> int:
    """Query the VLM for every uncached image of a split. Returns calls made."""
    import groq
    import crop_identify as ci

    client = ci.get_client()
    todo = [p for p in list_images(args.data, split) if (split, p.name) not in cache]
    print(f"  {split}: {len(todo)} images to query ({len(cache)} cached in total)")
    made = 0
    with open(args.out / "vlm_cache.jsonl", "a", encoding="utf-8") as f:
        for p in todo:
            if budget[0] <= 0:
                print("  [!] call budget reached — rerun later to resume")
                break
            for attempt in range(6):
                try:
                    raw = ci.query_vlm(client, str(p))
                    break
                except groq.RateLimitError as e:
                    retry = e.response.headers.get("retry-after") if e.response is not None else None
                    wait = float(retry) if retry else 2 ** attempt * 5
                    if wait > 600:
                        print(f"  [!] daily quota exhausted (retry-after {wait:.0f}s) — rerun later")
                        budget[0] = 0
                        return made
                    print(f"  429, sleeping {wait:.0f}s")
                    time.sleep(wait)
                except groq.APIStatusError as e:
                    raw = f"__error__ {e.status_code}"
                    break
            else:
                continue
            parsed = ci.parse_response(raw)
            r = {"split": split, "file": p.name, "gt": crop_of(p.name), "raw": raw, **parsed}
            cache[(split, p.name)] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            made += 1
            budget[0] -= 1
            if made % 50 == 0:
                print(f"    {made}/{len(todo)}")
            time.sleep(args.vlm_sleep)
    return made


def vlm_table(args, split: str, cache: dict) -> list[dict] | None:
    """Join VLM answers with YOLO image-level labels for one split."""
    imgs = list_images(args.data, split)
    if not all((split, p.name) in cache for p in imgs):
        return None
    preds = predict_split(args, split)
    rows = []
    for p in imgs:
        v = cache[(split, p.name)]
        y, yconf = image_label(preds[p.name])
        rows.append({"file": p.name, "gt": crop_of(p.name), "vlm": v["crop"], "vlm_conf": v["confidence"],
                     "yolo": y, "yolo_conf": round(yconf, 4)})
    return rows


def step_vlm(args, res: Results):
    if not os.environ.get("GROQ_API_KEY"):
        cache = load_vlm_cache(args.out / "vlm_cache.jsonl")
        if not cache:
            raise RuntimeError("GROQ_API_KEY is not set and there is no vlm_cache.jsonl")
        print("  GROQ_API_KEY not set — using cached answers only")
    else:
        cache = load_vlm_cache(args.out / "vlm_cache.jsonl")
        budget = [args.vlm_max_calls]
        for split in args.vlm_splits:
            run_vlm(args, split, cache, budget)

    for split in args.vlm_splits:
        rows = vlm_table(args, split, cache)
        if rows is None:
            n = sum(1 for k in cache if k[0] == split)
            print(f"  {split}: only {n} images answered so far — metrics after the split is complete")
            continue
        write_csv(args.out / f"vlm_vs_yolo_{split}.csv", rows)
        src = f"vlm_vs_yolo_{split}.csv"
        n = len(rows)
        acc = lambda key, rs: sum(r[key] == r["gt"] for r in rs) / max(len(rs), 1)
        agree = [r for r in rows if r["vlm"] == r["yolo"]]
        disagree = [r for r in rows if r["vlm"] != r["yolo"]]
        stats = {
            "n": n,
            "vlm_acc": acc("vlm", rows),
            "vlm_unknown": sum(r["vlm"] == "unknown" for r in rows),
            "yolo_acc": acc("yolo", rows),
            "yolo_none": sum(r["yolo"] == "none" for r in rows),
            "agreement": len(agree) / n,
            "acc_when_agree": acc("vlm", agree),
            "n_disagree": len(disagree),
            "vlm_acc_when_disagree": acc("vlm", disagree),
            "yolo_acc_when_disagree": acc("yolo", disagree),
        }
        for k, v in stats.items():
            res.add("vlm", f"{split}_{k}", round(v, 4) if isinstance(v, float) else v, src)

        # Confusion matrices (rows = ground truth)
        for model in ("vlm", "yolo"):
            cols = CLASSES + (["unknown"] if model == "vlm" else ["none"])
            cm = [{"gt": g, **{c: sum(1 for r in rows if r["gt"] == g and r[model] == c) for c in cols}}
                  for g in CLASSES]
            write_csv(args.out / f"{model}_confusion_{split}.csv", cm)
            for g in CLASSES:
                rg = [r for r in rows if r["gt"] == g]
                res.add("vlm", f"{split}_{model}_acc_{g}", round(acc(model, rg), 4), f"{model}_confusion_{split}.csv")

        print(f"\n  [{split}] n={n}  VLM acc {stats['vlm_acc']:.3f} (unknown {stats['vlm_unknown']})  "
              f"YOLO acc {stats['yolo_acc']:.3f} (none {stats['yolo_none']})")
        print(f"  agreement {stats['agreement']:.3f}  acc|agree {stats['acc_when_agree']:.3f}  "
              f"disagree n={len(disagree)}: VLM {stats['vlm_acc_when_disagree']:.3f} / "
              f"YOLO {stats['yolo_acc_when_disagree']:.3f}")


# ─── 5. Fusion rule ───────────────────────────────────────────────────────────

def fuse(r: dict, tau: float) -> str:
    """YOLO's label when its top confidence is at least tau, otherwise the VLM's."""
    return r["yolo"] if r["yolo"] != "none" and r["yolo_conf"] >= tau else r["vlm"]


def step_fusion(args, res: Results):
    tables = {}
    for split in ("val", "test"):
        f = args.out / f"vlm_vs_yolo_{split}.csv"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                tables[split] = [{**r, "yolo_conf": float(r["yolo_conf"])} for r in csv.DictReader(fh)]
    if "test" not in tables:
        raise RuntimeError("run the vlm step on the test split first")

    acc = lambda rs, tau: sum(fuse(r, tau) == r["gt"] for r in rs) / len(rs)
    rows = []
    for tau in TAUS:
        row = {"tau": tau}
        for split, rs in tables.items():
            row[f"{split}_acc"] = round(acc(rs, tau), 4)
            row[f"{split}_vlm_share"] = round(sum(fuse(r, tau) == r["vlm"] and r["vlm"] != r["yolo"]
                                                  for r in rs) / len(rs), 4)
        rows.append(row)
    write_csv(args.out / "fusion.csv", rows)

    pick_split = "val" if "val" in tables else "test"
    best = max(rows, key=lambda r: (r[f"{pick_split}_acc"], -r["tau"]))
    res.add("fusion", "tau_selected_on", pick_split, "fusion.csv")
    res.add("fusion", "best_tau", best["tau"], "fusion.csv")
    res.add("fusion", "test_acc_at_best_tau", best["test_acc"], "fusion.csv")
    for r in rows:
        for k, v in r.items():
            if k != "tau":
                res.add("fusion", f"tau{r['tau']}_{k}", v, "fusion.csv")

    print(f"\n  {'tau':>5s} " + " ".join(f"{k:>14s}" for k in rows[0] if k != "tau"))
    for r in rows:
        print(f"  {r['tau']:5.2f} " + " ".join(f"{v:14.4f}" for k, v in r.items() if k != "tau"))
    print(f"  best tau on {pick_split}: {best['tau']}  → test acc {best['test_acc']:.4f}")
    if pick_split == "test":
        print("  [!] tau chosen on test (no val VLM answers) — report as optimistic")


# ─── 6. Edge latency ──────────────────────────────────────────────────────────

LATENCY_WORKER = r"""
import json, sys, time, numpy as np, onnxruntime as ort, psutil, os
path, warm, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
so = ort.SessionOptions()
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
x = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
for _ in range(warm):
    sess.run(None, {inp.name: x})
t = []
for _ in range(runs):
    s = time.perf_counter(); sess.run(None, {inp.name: x}); t.append((time.perf_counter() - s) * 1000)
mi = psutil.Process().memory_info()
peak = getattr(mi, "peak_wset", None)
if peak is None:
    import resource; peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
t = np.array(t)
print(json.dumps({"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
                  "p95_ms": float(np.percentile(t, 95)), "fps": float(1000 / t.mean()),
                  "peak_rss_mb": peak / 2**20, "threads": os.cpu_count()}))
"""


def cpu_name() -> str:
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info()["brand_raw"]
    except Exception:
        return platform.processor()


def bench_onnx(path: Path, warm=20, runs=200) -> dict:
    """Benchmark in a fresh process so peak RSS only counts ONNX Runtime."""
    out = subprocess.run([sys.executable, "-c", LATENCY_WORKER, str(path), str(warm), str(runs)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def export_onnx(weights: Path, out: Path, tag: str) -> Path:
    from ultralytics import YOLO
    dst = out / f"{tag}_fp32.onnx"
    if not dst.exists():
        f = YOLO(str(weights)).export(format="onnx", imgsz=640, dynamic=False, simplify=True, batch=1)
        shutil.copy2(f, dst)
    return dst


def step_latency(args, res: Results):
    rows = []
    fp32 = export_onnx(args.weights, args.out, "yolov8n")
    variants = [("fp32", fp32)]
    if not args.no_int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            int8 = args.out / "yolov8n_int8.onnx"
            if not int8.exists():
                quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)
            variants.append(("int8_dynamic", int8))
        except Exception as e:
            print(f"  [!] INT8 quantization failed: {e}")

    for tag, path in variants:
        b = bench_onnx(path)
        row = {"model": "yolov8n", "precision": tag, "size_mb": round(path.stat().st_size / 2**20, 2),
               **{k: round(v, 2) for k, v in b.items()}, "cpu": cpu_name()}
        rows.append(row)

    # Accuracy cost of quantization on the test split (ONNX models via Ultralytics)
    if len(variants) > 1:
        from ultralytics import YOLO
        yaml_path = dataset_yaml(args.data, args.out)
        for row, (tag, path) in zip(rows, variants):
            try:
                m = YOLO(str(path), task="detect").val(data=str(yaml_path), split="test", imgsz=640, batch=1,
                                                       conf=VAL_CONF, iou=VAL_IOU, device="cpu", plots=False,
                                                       project=str(args.out / "onnx_val"), name=tag, exist_ok=True)
                row["test_mAP50"] = round(m.box.map50, 4)
                row["test_mAP50_95"] = round(m.box.map, 4)
            except Exception as e:
                print(f"  [!] ONNX val failed for {tag}: {e}")

    write_csv(args.out / "latency.csv", rows)
    print(f"\n  CPU: {cpu_name()}  (ONNX Runtime CPUExecutionProvider, 640x640, batch 1, 20 warm-up / 200 timed)")
    for r in rows:
        print(f"  {r['precision']:13s} mean {r['mean_ms']:7.2f} ms  p95 {r['p95_ms']:7.2f} ms  "
              f"{r['fps']:6.1f} FPS  peak RSS {r['peak_rss_mb']:6.1f} MB  size {r['size_mb']} MB  "
              f"mAP50 {r.get('test_mAP50', '-')}")
        for k in ("mean_ms", "p95_ms", "fps", "peak_rss_mb", "size_mb", "test_mAP50", "test_mAP50_95"):
            if k in r:
                res.add("latency", f"{r['precision']}_{k}", r[k], "latency.csv")
    res.add("latency", "cpu", cpu_name(), "latency.csv")
    import onnxruntime
    res.add("latency", "onnxruntime_version", onnxruntime.__version__, "latency.csv")


# ─── 7. PlantDoc cross-dataset check ──────────────────────────────────────────

PLANTDOC_MAP = {"corn": 2, "soyabean": 4, "soybean": 4}


def step_plantdoc(args, res: Results):
    """
    Expects the PlantDoc object-detection release (VOC XML next to each image),
    e.g. a clone of github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.
    Only corn and soybean objects are kept and mapped to our crop IDs.
    """
    import xml.etree.ElementTree as ET
    from ultralytics import YOLO

    if not args.plantdoc:
        raise RuntimeError("pass --plantdoc <path to PlantDoc-Object-Detection-Dataset>")
    src = args.plantdoc / "TEST" if (args.plantdoc / "TEST").exists() else args.plantdoc
    dst = args.out / "plantdoc_yolo"
    (dst / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "test").mkdir(parents=True, exist_ok=True)

    kept = Counter()
    for xml in sorted(src.glob("*.xml")):
        root = ET.parse(xml).getroot()
        fname = root.findtext("filename") or xml.stem + ".jpg"
        img = xml.with_name(fname) if xml.with_name(fname).exists() else xml.with_suffix(".jpg")
        if not img.exists():
            continue
        w = float(root.findtext("size/width") or 0)
        h = float(root.findtext("size/height") or 0)
        if not w or not h:
            from PIL import Image
            with Image.open(img) as im:
                w, h = im.size
        lines = []
        for obj in root.iter("object"):
            name = (obj.findtext("name") or "").lower()
            cid = next((v for k, v in PLANTDOC_MAP.items() if name.startswith(k)), None)
            if cid is None:
                continue
            bb = obj.find("bndbox")
            x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{cid} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            kept[CLASSES[cid]] += 1
        if lines:
            crop = CLASSES[int(lines[0].split()[0])]
            new = f"{crop}_{img.stem}"
            shutil.copy2(img, dst / "images" / "test" / (new + img.suffix.lower()))
            (dst / "labels" / "test" / (new + ".txt")).write_text("\n".join(lines))

    n_img = len(list((dst / "images" / "test").iterdir()))
    yaml_path = dataset_yaml(dst, args.out).rename(args.out / "plantdoc.yaml")
    m = YOLO(args.weights).val(data=str(yaml_path), split="test", imgsz=640, conf=VAL_CONF, iou=VAL_IOU,
                               device=args.device, project=str(args.out / "yolo_val"), name="plantdoc", exist_ok=True)
    rows = [{"class": "all", "images": n_img, "boxes": sum(kept.values()),
             "mAP50": round(m.box.map50, 4), "mAP50_95": round(m.box.map, 4)}]
    for i, c in enumerate(m.ap_class_index):
        _, _, ap50, ap = m.box.class_result(i)
        rows.append({"class": CLASSES[int(c)], "boxes": kept[CLASSES[int(c)]], "mAP50": round(ap50, 4), "mAP50_95": round(ap, 4)})
    write_csv(args.out / "plantdoc.csv", rows)
    for r in rows:
        print(f"  {r['class']:8s} boxes {r['boxes']:4d}  mAP50 {r['mAP50']:.3f}  mAP50-95 {r['mAP50_95']:.3f}")
        res.add("plantdoc", f"{r['class']}_mAP50", r["mAP50"], "plantdoc.csv")
        res.add("plantdoc", f"{r['class']}_mAP50_95", r["mAP50_95"], "plantdoc.csv")
    res.add("plantdoc", "images", n_img, "plantdoc.csv")


# ─── 8. n vs s ablation ───────────────────────────────────────────────────────

def step_ablation(args, res: Results):
    from ultralytics import YOLO

    if not args.weights_s:
        raise RuntimeError("pass --weights-s to compare YOLOv8n with YOLOv8s")
    yaml_path = dataset_yaml(args.data, args.out)
    rows = []
    for tag, w in (("yolov8n", args.weights), ("yolov8s", args.weights_s)):
        model = YOLO(str(w))
        from ultralytics.utils.torch_utils import get_flops
        n_params = sum(p.numel() for p in model.model.parameters())
        gflops = get_flops(model.model, imgsz=640)
        m = model.val(data=str(yaml_path), split="test", imgsz=640, conf=VAL_CONF, iou=VAL_IOU, plots=False,
                      device=args.device, project=str(args.out / "yolo_val"), name=f"ablation_{tag}", exist_ok=True)
        lat = bench_onnx(export_onnx(w, args.out, tag))
        rows.append({"model": tag, "params_M": round(n_params / 1e6, 2), "GFLOPs": round(gflops, 1),
                     "test_mAP50": round(m.box.map50, 4), "test_mAP50_95": round(m.box.map, 4),
                     "cpu_mean_ms": round(lat["mean_ms"], 2), "cpu_p95_ms": round(lat["p95_ms"], 2)})
    write_csv(args.out / "ablation.csv", rows)
    for r in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
        for k, v in r.items():
            if k != "model":
                res.add("ablation", f"{r['model']}_{k}", v, "ablation.csv")


# ─── Summary ──────────────────────────────────────────────────────────────────

def write_summary(res: Results, status: dict):
    lines = ["# Experiment summary", "",
             "Every number used in the paper, with the file in `experiments/out/` it came from.", "",
             "## Step status", "", "| step | status |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in status.items()]
    for step, vals in res.data.items():
        lines += ["", f"## {step}", "", "| key | value | source |", "|---|---|---|"]
        lines += [f"| {k} | {v['value']} | {v['source']} |" for k, v in vals.items()]
    (res.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


STEPS = {
    "dedup":    (step_dedup,    False),
    "stats":    (step_stats,    False),
    "detect":   (step_detect,   True),
    "vlm":      (step_vlm,      True),
    "fusion":   (step_fusion,   False),
    "latency":  (step_latency,  True),
    "plantdoc": (step_plantdoc, True),
    "ablation": (step_ablation, True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "dataset", help="merged dataset/ folder")
    ap.add_argument("--weights", type=Path, default=None, help="YOLOv8n best.pt")
    ap.add_argument("--weights-s", type=Path, default=None, help="YOLOv8s best.pt (ablation)")
    ap.add_argument("--sources", type=Path, default=ROOT, help="folder holding the Roboflow exports")
    ap.add_argument("--plantdoc", type=Path, default=None, help="PlantDoc detection dataset folder")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "out")
    ap.add_argument("--steps", default="dedup,stats,detect,vlm,fusion,latency,plantdoc,ablation")
    ap.add_argument("--vlm-splits", default="test,val", help="splits to query the VLM on, in order")
    ap.add_argument("--vlm-max-calls", type=int, default=950, help="per run (free tier: 1,000/day)")
    ap.add_argument("--vlm-sleep", type=float, default=2.1, help="seconds between calls (30 req/min)")
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--device", default="cpu", help="'cpu' or a CUDA index for val")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")       # Windows consoles default to cp1252
    args.vlm_splits = [s for s in args.vlm_splits.split(",") if s]
    args.out.mkdir(parents=True, exist_ok=True)

    res = Results(args.out)
    status = {}
    for name in args.steps.split(","):
        fn, needs_weights = STEPS[name]
        print(f"\n━━ {name} " + "━" * (60 - len(name)))
        if needs_weights and not (args.weights and args.weights.exists()):
            status[name] = "skipped (no --weights)"
            print(f"  skipped: needs --weights")
            continue
        t0 = time.time()
        try:
            fn(args, res)
            status[name] = f"ok ({time.time() - t0:.0f}s)"
        except Exception as e:
            status[name] = f"failed: {e}"
            traceback.print_exc()
        res.save()

    write_summary(res, status)
    print("\n━━ summary " + "━" * 52)
    for k, v in status.items():
        print(f"  {k:9s} {v}")
    print(f"\n  numbers → {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
