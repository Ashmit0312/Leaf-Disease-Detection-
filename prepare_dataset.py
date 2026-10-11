"""
Dataset Preparation Script for Crop Detection
=============================================
Creates the required folder structure and provides guidance for
collecting/annotating images of:
  - Sugarcane
  - Wheat
  - Maize (Corn)
  - Rice (Paddy)
  - Soybean

Run this FIRST before training.
"""

import os

# ─── Create Folder Structure ──────────────────────────────────────────────────

SPLITS = ["train", "val", "test"]
ROOTS  = ["images", "labels"]

for root in ROOTS:
    for split in SPLITS:
        path = os.path.join("dataset", root, split)
        os.makedirs(path, exist_ok=True)
        print(f"[✓] Created {path}")

print("\n✅ Folder structure ready.\n")


# ─── Class Map ────────────────────────────────────────────────────────────────

CLASS_MAP = {
    0: "sugarcane",
    1: "wheat",
    2: "maize",
    3: "rice",
    4: "soybean",
}

print("Class IDs:")
for k, v in CLASS_MAP.items():
    print(f"  {k} → {v}")

print("""
─────────────────────────────────────────────────────────────────
YOLO Label Format (per bounding box, one line per object):
  <class_id> <x_center> <y_center> <width> <height>
  All values normalized to [0, 1] relative to image size.
─────────────────────────────────────────────────────────────────

Recommended Dataset Size:
  ≥ 300 images per class for good results (1500+ total)
  Train : Val : Test = 70% : 20% : 10%

Free Image Sources:
  • PlantVillage dataset (Kaggle): plant disease + crop images
  • iNaturalist API: field crop observations
  • Roboflow Universe: search "sugarcane", "maize", etc.
  • ICRISAT OpenData: gram/soybean crop images
  • ICAR Kharif/Rabi season datasets

Annotation Tools (free):
  • Roboflow (web, auto-exports YOLO format): roboflow.com
  • LabelImg (local desktop tool): pip install labelImg
  • CVAT (web): cvat.ai

Quick Start with Roboflow:
  1. Upload images to Roboflow project
  2. Annotate bounding boxes for each crop
  3. Export → YOLOv8 format
  4. Download and place in dataset/images/ and dataset/labels/

─────────────────────────────────────────────────────────────────
""")


# ─── Optional: Download from Roboflow ─────────────────────────────────────────

ROBOFLOW_EXAMPLE = """
# Uncomment and fill in your Roboflow API key + workspace to auto-download:

# pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# project = rf.workspace("your-workspace").project("crop-detection")
# dataset = project.version(1).download("yolov8")
"""

print(ROBOFLOW_EXAMPLE)
