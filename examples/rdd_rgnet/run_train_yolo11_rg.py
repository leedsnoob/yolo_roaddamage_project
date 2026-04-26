import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11-rg.yaml"
DEFAULT_WEIGHTS = ROOT / "yolo11n.pt"


if __name__ == "__main__":
    cfg = Path(os.environ.get("YOLO_MODEL_CFG", DEFAULT_CFG))
    weights = Path(os.environ.get("YOLO_PRETRAINED_WEIGHTS", DEFAULT_WEIGHTS))
    model = YOLO(str(cfg)).load(str(weights if weights.exists() else "yolo11n.pt"))
    model.train(
        data="rdd2022.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device="0",
        optimizer="SGD",
        lr0=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        plots=True,
    )
