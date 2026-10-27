# Revision changelog: reviewer point → what was done

Status key: **Done** = in the paper and backed by a script output; **Ready** = code written and smoke-tested, waiting on retrained weights / API key; **Deferred** = out of scope for this revision, stated in the paper.

Every number added to the paper comes from `experiments/out/summary.md` (original split, `dataset_v2`) or `experiments/out_clean/summary.md` (clean split, `dataset_clean`).

| # | Reviewer point | Status | What was done |
|---|---|---|---|
| 1 | No dedup / provenance audit; possible leakage across splits | **Done** | `run_all.py --steps dedup`: MD5, 64-bit pHash (≤4), all 8 flips/90° rotations for val/test, Roboflow `<orig>.rf.<hash>` source groups. Original split: **223/1,098 val (20.3%)** and **138/551 test (25.0%)** leaked; 40 source groups span splits; 0 cross-crop duplicates. Manual check of 12 random distance-4 pairs: 9 clear duplicates, 3 ambiguous single-leaf-on-white shots (audit errs toward flagging). `clean_split.py` builds a group-aware split (union–find; groups touching train → train, val+test → val): train 10,053 / val 880 / test 413. Re-audit: **0 cross-split duplicates**. New §III-B and Table II. |
| 2 | Exact Roboflow datasets, URLs, licences not listed | **Done** | Read from each export's `data.yaml`; all five are CC BY 4.0. Table I + five dataset references [20]–[24]. Maize subsampling (1,200 of 10,000, seed 42, proportional to splits) stated. |
| 3 | Rejected polygon labels | **Done** | `merge_datasets.py` now converts polygon rows to clipped bounding boxes (3,020 rows in 2,709 files) instead of letting Ultralytics reject 53 files. |
| 4 | Release a cleaned manifest with image hashes | **Done** | `experiments/out_clean/manifest.csv` (split, file, crop, MD5, pHash). |
| 5 | Boxes per crop / per source | **Done** | `class_stats.csv`, `class_totals.csv`; Table I gives images, boxes and median box area per crop. |
| 6 | Mixed leaf- vs lesion-level boxes | **Done (quantified)** / Deferred (fix) | Median box area: maize 98.3% (1 box/image) vs soybean 0.1% (8.0 boxes/image, 74.5% of boxes <1% of image). Discussion now notes per-crop AP is confounded with annotation style and box scale can act as a crop shortcut; LAT-style label transfer named as remedy. Leaf-only re-annotation baseline deferred. |
| 7 | "Lower bound" claim unsupported | **Done** | Wording removed. Text now says the curves only show the run was stopped before convergence; the full 100-epoch run on the clean split settles it. First-run numbers are labelled optimistic (leaky split, rejected labels, 51-image-larger earlier copy). |
| 8 | No test-set results, per-class AP, confusion matrix, error analysis | **Ready** | `--steps detect`: val + test P/R/mAP@50/mAP@50-95, per-crop rows, `*_confusion_matrix_normalized.png`, `*_BoxPR_curve.png`, per-image matching at IoU 0.5 → `test_per_image.csv`, 10 worst cases drawn to `worst_cases/`. Smoke-tested with a throwaway 1-epoch model. |
| 9 | NMS / confidence thresholds | **Done** | Stated in §III-H: conf 0.001, NMS IoU 0.7 (Ultralytics val defaults); image-level labels use conf 0.25. |
| 10 | VLM not quantitatively evaluated; no conflict-resolution rule | **Done (rule)** / **Ready (numbers)** | Fusion rule defined in §III-G and Fig. 1: detector label if top confidence ≥ τ, else VLM; τ ∈ {0.25, 0.4, 0.5, 0.6, 0.75} chosen on val. `--steps vlm,fusion` reports VLM accuracy, unknown count, per-crop confusion, YOLO image-level accuracy, agreement, accuracy when agreeing/disagreeing, and fused accuracy. Caches every answer, sleeps 2.1 s, backs off on 429, stops at a daily budget. Join and fusion logic smoke-tested with a fake cache; real run needs `GROQ_API_KEY`. VLM section updated to match the current prompt (strict JSON, 0–100 confidence, temperature 0.1, 100 tokens). |
| 11 | No CPU latency / memory, no quantization | **Ready** | `--steps latency`: static 640×640 ONNX, ONNX Runtime CPU EP, 20 warm-up + 200 timed runs in a separate process (so peak RSS counts only ONNX Runtime), mean/p50/p95 ms, FPS, peak RSS, CPU name; INT8 dynamic quantization with test mAP of both. |
| 12 | No ablation over model scale | **Ready** | `retrain_colab.ipynb` trains YOLOv8s with identical settings; `--steps ablation` reports params, GFLOPs, test mAP, CPU latency. |
| 13 | No cross-dataset test (PlantDoc) | **Ready** / Deferred | `--steps plantdoc --plantdoc <path>` maps PlantDoc corn → maize (2), soybean → 4. Needs the PlantDoc dataset downloaded (not done without your OK) and weights. Listed as not yet run in §V. |
| 14 | Missing open-vocabulary and dataset-merging related work | **Done** | Added CLIP, OWL-ViT, Grounding DINO, YOLO-World, Simple Multi-Dataset Detection (UniDet), LAT, GWHD; each verified by web search (venue, year, pages/DOI). New paragraphs explain why image-level VLM classification was chosen. |
| 15 | Domain-shift benchmarks not discussed | **Done** | GWHD + PlantDoc cited in Related Work and Limitations. |
| 16 | Cloud VLM impractical in low connectivity | **Done** | New Limitations paragraph; fusion rule limits calls; on-device open-vocabulary detector / small local VLM as future work. |
| 17 | Baselines: per-crop vs unified, no-mosaic, N-EIoU, two-stage | Deferred | Out of scope for the GPU budget before the 30 Nov deadline. Not claimed. |

## Code changes

- `merge_datasets.py`: polygon → box conversion, `--out` flag, fixed wheat folder name casing (broke on Linux/Colab).
- `crop_identify.py`: split into `get_client` / `query_vlm` / `parse_response` so evaluation reuses the exact prompt and settings; tolerant JSON parsing (code fences); unknown answers get confidence 0.
- `train.py`: `--project`, `--name`, `--batch`, `--save-period` (default 5), auto-resume from `last.pt`, `seed=0`, correct `best.pt` path (Ultralytics nests relative projects under `runs/detect/`), `CROP_DATASET` env var.
- `download.py`: the Roboflow API key was hard-coded; it now reads `ROBOFLOW_API_KEY`. **The old key is in local commit `634e9ee5` — rotate it in Roboflow before pushing.**
- New: `experiments/run_all.py`, `experiments/clean_split.py`, `experiments/retrain_colab.ipynb`.

## What is still needed to finish the Results section

1. Zip the contents of `dataset_clean/` as `dataset_clean.zip`, upload to `MyDrive/LDD/`, run `experiments/retrain_colab.ipynb` (YOLOv8n ≈ 5–6 h on a T4; YOLOv8s optional).
2. Put `best.pt` in `weights/`, set `GROQ_API_KEY`, then run:
   `python experiments/run_all.py --data dataset_clean --weights weights/v8n_best.pt --out experiments/out_clean --steps detect,vlm,fusion,latency`
   (VLM needs about two days of free-tier quota for test + val; rerun to resume.)
3. Then the Results section gets: test/per-crop AP table, confusion matrix, error analysis, VLM + fusion table, latency table, updated curves and abstract.
