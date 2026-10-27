# 🌾 Leaf Disease Detection Using Computer Vision

Multi-crop detection for **Sugarcane, Wheat, Maize, Rice and Soybean** using
**YOLOv8** (bounding-box detection) plus a zero-shot **vision-language model**
(Groq `llama-4-scout-17b-16e-instruct`) for crop identification.

Project of the Dept. of Information Technology, Dwarkadas J. Sanghvi College of Engineering (DJSCE), University of Mumbai, 2025-26.
**Team:** Ashmit Janjirkar, Hitansh Gada, Aryan Gada, Urav Jain  
**Guide:** Dr. Monika Mangla

> **Status:** work in progress (~60%). Dataset pipeline, YOLO training and the Groq crop identifier work. FastAPI backend, Flutter app and Grad-CAM are pending.

## How it works

| Stage | Model | Needs data? |
|---|---|---|
| Crop identification | Groq Llama-4-Scout (zero-shot) | No |
| Detection | YOLOv8 (trained by us) | Yes, bounding-box annotated |
| Deployment | ONNX export | - |

| Class ID | Crop |
|---|---|
| 0 | sugarcane |
| 1 | wheat |
| 2 | maize |
| 3 | rice |
| 4 | soybean |

## Repo contents

```
├── prepare_dataset.py    # creates empty dataset/ folders
├── merge_datasets.py     # merges per-crop Roboflow exports into one YOLO dataset
├── train.py              # train / val / predict / export (ONNX)
├── crop_identify.py      # zero-shot crop ID via Groq
├── notebooks/
│   ├── Crop_Detection_Colab.ipynb   # dataset read from Google Drive
│   └── Crop_Detection_Local.ipynb   # dataset read from your laptop
├── requirements.txt
└── .gitignore
```

Datasets, weights (`.pt`, `.onnx`) and `runs/` are **not** in the repo.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get the data

Download one **YOLOv8-format** dataset per crop from [Roboflow Universe](https://universe.roboflow.com).
Roboflow exports `train/images`, `valid/images`, `test/images`; `merge_datasets.py` handles this layout.

Edit the folder names in `CROP_SOURCES` at the top of `merge_datasets.py`, then:

```bash
python merge_datasets.py
```

This writes `dataset/` (`images/`, `labels/`, `crop_dataset.yaml`) and rewrites every label's class ID to 0-4.

## 2. Train

```bash
python train.py --mode train --size n --epochs 50 --device 0   # --device cpu if no GPU
python train.py --mode val --weights runs/crop_detect/exp1/weights/best.pt
python train.py --mode predict --weights runs/crop_detect/exp1/weights/best.pt --source image.jpg
python train.py --mode export --weights runs/crop_detect/exp1/weights/best.pt
```

Or use a notebook. In `Crop_Detection_Local.ipynb`, set `DATASET_ROOT` in the first dataset cell to your local path.

## 3. Crop identification (Groq)

Get a free key at [console.groq.com](https://console.groq.com) (no card needed).

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_key"
# macOS / Linux
export GROQ_API_KEY="your_key"

python crop_identify.py --image path/to/leaf.jpg
```

**Never commit API keys.** The code reads `GROQ_API_KEY` from the environment.

## Roadmap

- [ ] Per-disease classes per crop
- [ ] FastAPI inference backend
- [ ] Flutter mobile app (offline ONNX)
- [ ] Grad-CAM explainability
- [ ] Field testing in Maharashtra

## Acknowledgements

Ultralytics YOLOv8, Roboflow Universe datasets, Groq / Meta Llama 4.
---

## 🔬 Paper Revision Experiments

```bash
# 1. Re-merge sources (polygon labels → boxes) into a fresh folder
python merge_datasets.py --out dataset_v2

# 2. Leakage audit + per-crop stats (no weights needed)
python experiments/run_all.py --data dataset_v2 --steps dedup,stats --out experiments/out

# 3. Build the group-aware split with no cross-split duplicates
python experiments/clean_split.py --data dataset_v2 --report experiments/out/dedup_report.csv --out dataset_clean

# 4. Retrain on Colab with experiments/retrain_colab.ipynb, then evaluate locally
#    (VLM step reads GROQ_API_KEY from the environment and resumes from its cache)
python experiments/run_all.py --data dataset_clean --weights weights/v8n_best.pt --out experiments/out_clean --steps detect,vlm,fusion,latency
```

Every number used in the paper is listed with its source file in `experiments/out*/summary.md`.
