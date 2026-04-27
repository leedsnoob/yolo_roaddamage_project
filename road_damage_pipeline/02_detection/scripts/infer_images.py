#!/usr/bin/env python3
"""Run the packaged detector on the small non-drone, non-empty RDD image subset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

cv2 = None


def require_cv2():
    global cv2
    if cv2 is None:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: opencv-python. Install road_damage_pipeline/requirements.txt first.") from exc
        cv2 = cv2_module
    return cv2


MODULE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = MODULE_ROOT.parent


def default_repo_root() -> Path:
    if (PIPELINE_ROOT.parent / "ultralytics").exists():
        return PIPELINE_ROOT.parent
    return PIPELINE_ROOT.parent / "ultralytics_yolo11_final"


DEFAULT_REPO_ROOT = default_repo_root()
DEFAULT_IMAGE_DIR = MODULE_ROOT / "samples" / "images" / "raw"
DEFAULT_LABEL_DIR = MODULE_ROOT / "samples" / "images" / "labels"
DEFAULT_WEIGHTS = MODULE_ROOT / "weights" / "yolo11s_original_nondrone_noempty_best.pt"
DEFAULT_OUTPUT_ROOT = MODULE_ROOT / "outputs" / "sample_inference"

CLASS_ID_TO_CODE = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_ID_TO_NAME = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}
CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (0, 255, 0),
    3: (0, 128, 255),
}


def resolve_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), DEFAULT_REPO_ROOT, DEFAULT_REPO_ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def resolve_repo_root(value: str | Path | None) -> Path:
    if not value:
        return DEFAULT_REPO_ROOT
    path = resolve_path(value)
    if path and path.exists():
        return path
    if Path(value).name == "ultralytics_yolo11_final" and DEFAULT_REPO_ROOT.exists():
        return DEFAULT_REPO_ROOT
    return path or DEFAULT_REPO_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def draw_box(image, xyxy: list[float], class_id: int, label: str) -> None:
    cv2_mod = require_cv2()
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    cv2_mod.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2_mod.getTextSize(label, cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(y1 - 8, th + 4)
    cv2_mod.rectangle(image, (x1, y_text - th - 5), (x1 + tw + 8, y_text + 3), color, -1)
    cv2_mod.putText(image, label, (x1 + 4, y_text - 2), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2_mod.LINE_AA)


def validate_sample_labels(image_dir: Path, label_dir: Path) -> dict:
    image_paths = sorted(image_dir.glob("*.jpg"))
    empty_labels = []
    drone_images = []
    missing_labels = []
    for image_path in image_paths:
        if "Drone" in image_path.name:
            drone_images.append(image_path.name)
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        if not label_path.read_text(encoding="utf-8").strip():
            empty_labels.append(label_path.name)
    return {
        "num_images": len(image_paths),
        "drone_images": drone_images,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "is_non_drone_non_empty": not drone_images and not missing_labels and not empty_labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run packaged YOLO detector on sample RDD images.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.ones(1, device="mps")
                return "mps"
            except Exception:
                return "cpu"
    except Exception:
        return "cpu"
    return "cpu"


def setup_ultralytics(repo_root: Path):
    cache_dir = ensure_dir(MODULE_ROOT / ".ultralytics_cache")
    os.environ["YOLO_CONFIG_DIR"] = str(cache_dir)
    if str(repo_root.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_root.resolve()))
    from ultralytics import YOLO

    return YOLO


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    weights = resolve_path(args.weights)
    image_dir = resolve_path(args.image_dir)
    label_dir = resolve_path(args.label_dir)
    output_root = resolve_path(args.output_root) or DEFAULT_OUTPUT_ROOT

    if not repo_root.exists():
        raise FileNotFoundError(f"YOLO repo root not found: {repo_root}")
    if not weights or not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not image_dir or not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir or not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    validation = validate_sample_labels(image_dir, label_dir)
    if not validation["is_non_drone_non_empty"]:
        raise RuntimeError(f"Sample image set is not clean non-drone non-empty: {validation}")

    YOLO = setup_ultralytics(repo_root)
    model = YOLO(str(weights))
    cv2_mod = require_cv2()
    device = resolve_device(args.device)

    image_paths = sorted(image_dir.glob("*.jpg"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    pred_dir = ensure_dir(output_root / "predicted_images")
    rows = []
    counts = Counter()
    for image_path in image_paths:
        image = cv2_mod.imread(str(image_path))
        if image is None:
            continue
        result = model.predict(
            image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for det_idx, (xyxy, conf, cls_id) in enumerate(
                zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.int().cpu().tolist())
            ):
                cls_id = int(cls_id)
                code = CLASS_ID_TO_CODE.get(cls_id, f"class_{cls_id}")
                counts[code] += 1
                xyxy = [round(float(v), 2) for v in xyxy]
                rows.append(
                    {
                        "image_name": image_path.name,
                        "detection_id": f"{image_path.stem}-{det_idx}",
                        "class_id": cls_id,
                        "class_code": code,
                        "class_name": CLASS_ID_TO_NAME.get(cls_id, str(cls_id)),
                        "confidence": round(float(conf), 4),
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                    }
                )
                draw_box(image, xyxy, cls_id, f"{code} {float(conf):.2f}")
        cv2_mod.imwrite(str(pred_dir / f"{image_path.stem}_pred.jpg"), image)

    ensure_dir(output_root)
    detections_csv = output_root / "detections.csv"
    with detections_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image_name", "detection_id", "class_id", "class_code", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "image_dir": str(image_dir),
        "weights": str(weights),
        "repo_root": str(repo_root),
        "device_actual": device,
        "sample_validation": validation,
        "num_images_processed": len(image_paths),
        "total_detections": len(rows),
        "counts_by_class": dict(counts),
        "detections_csv": str(detections_csv),
        "predicted_images_dir": str(pred_dir),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
