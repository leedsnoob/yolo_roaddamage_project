import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
DEFAULT_ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics_yolo11_final"
DEFAULT_WEIGHTS = PIPELINE_ROOT / "fastsam" / "weights" / "FastSAM-s.pt"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "fastsam_road_demo"


def parse_args():
    parser = argparse.ArgumentParser(description="Run FastSAM road segmentation demo for PIDNet comparison.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="FastSAM checkpoint path.")
    parser.add_argument("--ultralytics-root", default=str(DEFAULT_ULTRALYTICS_ROOT), help="Local ultralytics fork root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for visualization outputs.")
    parser.add_argument("--mode", choices=["segment_all", "text_road", "both"], default="both")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, mps, etc.")
    return parser.parse_args()


def resolve_device(device: str):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_imports(ultralytics_root: Path):
    if str(ultralytics_root) not in sys.path:
        sys.path.insert(0, str(ultralytics_root))


def collect_masks(result, height: int, width: int):
    if result.masks is None:
        return np.zeros((height, width), dtype=np.uint8), 0

    masks = result.masks.data.detach().cpu().numpy()
    union = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        union[mask > 0.5] = 1
    return union, int(len(masks))


def render_overlay(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 180, 255)):
    overlay = image_bgr.copy()
    tint = np.zeros_like(image_bgr)
    tint[:, :] = color
    active = mask > 0
    overlay[active] = cv2.addWeighted(image_bgr[active], 0.55, tint[active], 0.45, 0)
    return overlay


def add_title(image_bgr: np.ndarray, title: str):
    canvas = cv2.copyMakeBorder(image_bgr, 38, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def run_prediction(model, image_path: Path, args, texts=None):
    kwargs = dict(
        source=str(image_path),
        device=resolve_device(args.device),
        retina_masks=False,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )
    if texts is None:
        return model.predict(**kwargs)[0]
    return model.predict(**kwargs, texts=texts)[0]


def main():
    args = parse_args()
    ultralytics_root = Path(args.ultralytics_root).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_imports(ultralytics_root)
    from ultralytics import FastSAM

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = FastSAM(str(weights))
    height, width = image_bgr.shape[:2]
    panels = [("Original", image_bgr)]
    summary = {
        "image": str(image_path),
        "weights": str(weights),
        "mode": args.mode,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": resolve_device(args.device),
    }

    if args.mode in {"segment_all", "both"}:
        result = run_prediction(model, image_path, args)
        mask, count = collect_masks(result, height, width)
        panels.append((f"FastSAM segment-all ({count})", render_overlay(image_bgr, mask)))
        summary["segment_all_masks"] = count

    if args.mode in {"text_road", "both"}:
        result = run_prediction(model, image_path, args, texts=["road"])
        mask, count = collect_masks(result, height, width)
        panels.append((f"FastSAM text=road ({count})", render_overlay(image_bgr, mask, color=(0, 255, 0))))
        summary["text_road_masks"] = count

    board = np.concatenate([add_title(panel, title) for title, panel in panels], axis=1)
    out_image = output_dir / f"{image_path.stem}_fastsam_{args.mode}.jpg"
    out_json = output_dir / f"{image_path.stem}_fastsam_{args.mode}.json"
    cv2.imwrite(str(out_image), board)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"visual": str(out_image), "summary": str(out_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
