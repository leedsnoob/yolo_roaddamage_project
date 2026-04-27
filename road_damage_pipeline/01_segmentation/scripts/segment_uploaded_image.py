#!/usr/bin/env python3
"""Run live PIDNet road segmentation for one uploaded image.

This script is intentionally independent from the packaged comparison assets.
It produces per-job visual evidence for the image the user actually uploaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


MODULE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = MODULE_ROOT.parent
REPO_ROOT = PIPELINE_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent if REPO_ROOT.name == "ultralytics_yolo11_final" else PIPELINE_ROOT.parent

PIDNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIDNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def candidate_paths(*relative_parts: str) -> list[Path]:
    return [
        MODULE_ROOT.joinpath(*relative_parts),
        WORKSPACE_ROOT.joinpath(*relative_parts),
        WORKSPACE_ROOT / "segmentation_pipeline" / Path(*relative_parts),
    ]


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def default_pidnet_root() -> Path | None:
    return first_existing(
        [
            MODULE_ROOT / "pidnet" / "source",
            WORKSPACE_ROOT / "segmentation_pipeline" / "pidnet" / "source",
        ]
    )


def default_pidnet_weights() -> Path | None:
    return first_existing(
        [
            MODULE_ROOT / "weights" / "pidnet" / "PIDNet_L_Cityscapes_test.pt",
            WORKSPACE_ROOT / "segmentation_pipeline" / "pidnet" / "weights" / "PIDNet_L_Cityscapes_test.pt",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIDNet road segmentation on one uploaded image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pidnet-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--road-class-id", type=int, default=0)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.ones(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def setup_imports(pidnet_root: Path) -> None:
    if str(pidnet_root) not in sys.path:
        sys.path.insert(0, str(pidnet_root))


def load_pidnet(pidnet_root: Path, weights_path: Path, device: torch.device):
    setup_imports(pidnet_root)
    import models

    model = models.pidnet.get_pred_model("pidnet-l", 19)
    pretrained = torch.load(weights_path, map_location="cpu")
    pretrained = pretrained["state_dict"] if "state_dict" in pretrained else pretrained
    model_dict = model.state_dict()
    pretrained = {
        key[6:]: value
        for key, value in pretrained.items()
        if key.startswith("model.") and key[6:] in model_dict and model_dict[key[6:]].shape == value.shape
    }
    model_dict.update(pretrained)
    model.load_state_dict(model_dict, strict=False)
    model.to(device).eval()
    return model


def pidnet_input_transform(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.astype(np.float32)[:, :, ::-1] / 255.0
    image -= PIDNET_MEAN
    image /= PIDNET_STD
    return image.transpose((2, 0, 1)).copy()


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def predict(model, image_bgr: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(pidnet_input_transform(image_bgr)).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    aligned_height = int(np.ceil(height / 32.0) * 32)
    aligned_width = int(np.ceil(width / 32.0) * 32)
    if (aligned_height, aligned_width) != (height, width):
        tensor = F.interpolate(tensor, size=(aligned_height, aligned_width), mode="bilinear", align_corners=True)
    sync_if_needed(device)
    start = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    sync_if_needed(device)
    return pred, time.perf_counter() - start


def road_visuals(image_bgr: np.ndarray, road_mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = (road_mask > 0).astype(np.uint8)
    road_only = np.zeros_like(image_bgr)
    road_only[mask > 0] = image_bgr[mask > 0]

    mask_rgb = np.zeros_like(image_bgr)
    mask_rgb[mask > 0] = (80, 255, 90)
    overlay = cv2.addWeighted(image_bgr, 0.72, mask_rgb, 0.42, 0)
    overlay[mask == 0] = (overlay[mask == 0] * 0.42).astype(np.uint8)

    binary = np.full_like(image_bgr, 12)
    binary[mask > 0] = (80, 255, 90)
    return {"road_only": road_only, "road_overlay": overlay, "road_mask": binary}


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((42, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([image, bar])


def make_triptych(raw: np.ndarray, road_only: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    panels = [
        add_title(raw, "Original uploaded image"),
        add_title(road_only, "PIDNet road region"),
        add_title(overlay, "PIDNet road overlay"),
    ]
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.full((height - panel.shape[0], panel.shape[1], 3), 245, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    separator = np.full((height, 10, 3), 245, dtype=np.uint8)
    return np.hstack([padded[0], separator, padded[1], separator, padded[2]])


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pidnet_root = args.pidnet_root or default_pidnet_root()
    weights = args.weights or default_pidnet_weights()
    if pidnet_root is None or weights is None:
        summary = {
            "status": "skipped",
            "reason": "PIDNet source or checkpoint is missing.",
            "expected_pidnet_root": str(MODULE_ROOT / "pidnet" / "source"),
            "fallback_pidnet_root": str(WORKSPACE_ROOT / "segmentation_pipeline" / "pidnet" / "source"),
            "expected_weights": str(MODULE_ROOT / "weights" / "pidnet" / "PIDNet_L_Cityscapes_test.pt"),
            "fallback_weights": str(WORKSPACE_ROOT / "segmentation_pipeline" / "pidnet" / "weights" / "PIDNet_L_Cityscapes_test.pt"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    device = resolve_device(args.device)
    model = load_pidnet(pidnet_root, weights, device)
    pred, elapsed = predict(model, image, device)
    road_mask = (pred == args.road_class_id).astype(np.uint8)
    visuals = road_visuals(image, road_mask)

    raw_path = output_dir / f"{args.image.stem}_raw.jpg"
    road_path = output_dir / f"{args.image.stem}_pidnet_road.jpg"
    overlay_path = output_dir / f"{args.image.stem}_pidnet_overlay.jpg"
    mask_path = output_dir / f"{args.image.stem}_pidnet_mask.jpg"
    triptych_path = output_dir / f"{args.image.stem}_pidnet_triptych.jpg"

    cv2.imwrite(str(raw_path), image)
    cv2.imwrite(str(road_path), visuals["road_only"])
    cv2.imwrite(str(overlay_path), visuals["road_overlay"])
    cv2.imwrite(str(mask_path), visuals["road_mask"])
    cv2.imwrite(str(triptych_path), make_triptych(image, visuals["road_only"], visuals["road_overlay"]))

    road_pixels = int(road_mask.sum())
    summary = {
        "status": "done",
        "mode": "live_uploaded_image",
        "image": str(args.image),
        "pidnet_root": str(pidnet_root),
        "weights": str(weights),
        "device": str(device),
        "road_class_id": args.road_class_id,
        "road_pixels": road_pixels,
        "road_ratio": road_pixels / int(road_mask.size),
        "runtime_s": round(elapsed, 6),
        "outputs": {
            "raw": str(raw_path),
            "road": str(road_path),
            "overlay": str(overlay_path),
            "mask": str(mask_path),
            "triptych": str(triptych_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
