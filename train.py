"""
Crop Detection using YOLOv8
Crops: Sugarcane, Wheat, Maize, Rice, Soybean

Requirements:
    pip install ultralytics torch torchvision pillow pyyaml

Dataset structure expected:
    dataset/
      images/
        train/   *.jpg
        val/     *.jpg
        test/    *.jpg
      labels/
        train/   *.txt   (YOLO format)
        val/     *.txt
        test/    *.txt
"""

from ultralytics import YOLO
import yaml, os

# ─── 1. Dataset Config ────────────────────────────────────────────────────────

DATASET_YAML = "crop_dataset.yaml"

dataset_config = {
    "path": os.path.abspath("dataset"),   # root dir
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",

    "nc": 5,                              # number of classes
    "names": {
        0: "sugarcane",
        1: "wheat",
        2: "maize",
        3: "rice",
        4: "soybean",
    }
}

with open(DATASET_YAML, "w") as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

print(f"[✓] Dataset YAML written → {DATASET_YAML}")


# ─── 2. Model + Training ──────────────────────────────────────────────────────

def train(
    model_size: str = "n",          # n | s | m | l | x
    epochs: int     = 100,
    img_size: int   = 640,
    batch: int      = 16,
    device: str     = "0",          # "0" for GPU, "cpu" for CPU
    project: str    = "runs/crop_detect",
    name: str       = "exp1",
):
    model_name = f"yolov8{model_size}.pt"
    print(f"\n[→] Loading pretrained {model_name} ...")
    model = YOLO(model_name)

    print(f"[→] Starting training for {epochs} epochs ...\n")
    results = model.train(
        data      = DATASET_YAML,
        epochs    = epochs,
        imgsz     = img_size,
        batch     = batch,
        device    = device,
        project   = project,
        name      = name,
        # Augmentation (helps with field variability)
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.5,
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.1,
        copy_paste= 0.1,
        degrees   = 10.0,
        translate = 0.1,
        scale     = 0.5,
        shear     = 2.0,
    )

    best_weights = f"{project}/{name}/weights/best.pt"
    print(f"\n[✓] Training complete. Best weights → {best_weights}")
    return best_weights


# ─── 3. Validation ────────────────────────────────────────────────────────────

def validate(weights: str, split: str = "val"):
    model = YOLO(weights)
    metrics = model.val(data=DATASET_YAML, split=split)
    print(f"\n[✓] Validation mAP@50   : {metrics.box.map50:.4f}")
    print(f"    Validation mAP@50-95 : {metrics.box.map:.4f}")
    return metrics


# ─── 4. Inference ─────────────────────────────────────────────────────────────

def predict(weights: str, source: str, conf: float = 0.25, save: bool = True):
    """
    source: path to image / folder / video / URL / 0 (webcam)
    """
    model = YOLO(weights)
    results = model.predict(
        source  = source,
        conf    = conf,
        save    = save,
        save_txt= True,
        project = "runs/predict",
    )
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            name   = dataset_config["names"][cls_id]
            conf_  = float(box.conf)
            xyxy   = box.xyxy[0].tolist()
            print(f"  Detected: {name:10s}  conf={conf_:.2f}  bbox={[round(v,1) for v in xyxy]}")
    return results


# ─── 5. Export (ONNX for deployment) ─────────────────────────────────────────

def export_onnx(weights: str):
    model = YOLO(weights)
    model.export(format="onnx", dynamic=True, simplify=True)
    print(f"[✓] Exported ONNX → {weights.replace('.pt','.onnx')}")


# ─── 6. Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop YOLO Pipeline")
    parser.add_argument("--mode",    choices=["train","val","predict","export"], default="train")
    parser.add_argument("--weights", default=None,  help="Path to .pt weights (for val/predict/export)")
    parser.add_argument("--source",  default=None,  help="Image/folder path for prediction")
    parser.add_argument("--size",    default="n",   help="YOLOv8 model size: n/s/m/l/x")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--device",  default="0")
    args = parser.parse_args()

    if args.mode == "train":
        w = train(model_size=args.size, epochs=args.epochs, device=args.device)
        validate(w)

    elif args.mode == "val":
        assert args.weights, "--weights required for val"
        validate(args.weights)

    elif args.mode == "predict":
        assert args.weights and args.source, "--weights and --source required"
        predict(args.weights, args.source)

    elif args.mode == "export":
        assert args.weights, "--weights required for export"
        export_onnx(args.weights)
