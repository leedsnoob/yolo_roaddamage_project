"""Build crop-only FastSAM segmentation-quality visuals for selected RDD GT boxes.

This script is intentionally separate from area estimation. It does not compute
square-meter areas. It only visualizes whether FastSAM can produce useful masks
when the input is the GT bbox crop and the text prompt is the detected damage.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MODULE_ROOT.parents[1]
SEGMENTATION_ROOT = WORKSPACE_ROOT / "segmentation_pipeline"
DEFAULT_IMAGE_ROOT = MODULE_ROOT / "samples" / "images" / "raw"
DEFAULT_BBOX_CSV = MODULE_ROOT / "samples" / "images" / "gt_bboxes_from_txt.csv"
DEFAULT_OUTPUT_ROOT = MODULE_ROOT / "assets" / "area_segmentation_quality_crop"
DEFAULT_FASTSAM_WEIGHTS = SEGMENTATION_ROOT / "fastsam" / "weights" / "FastSAM-s.pt"
DEFAULT_ULTRALYTICS_ROOT = WORKSPACE_ROOT / "ultralytics_yolo11_final"

CLASS_COLORS = {
    "D00": (245, 158, 11),
    "D10": (59, 130, 246),
    "D20": (16, 185, 129),
    "D40": (239, 68, 68),
}

FASTSAM_COLORS = [
    (0, 255, 255),
    (255, 200, 0),
    (0, 220, 120),
    (255, 120, 120),
    (120, 180, 255),
    (180, 120, 255),
    (255, 220, 120),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crop-only FastSAM segmentation quality visuals.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--bbox-csv", default=str(DEFAULT_BBOX_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--fastsam-weights", default=str(DEFAULT_FASTSAM_WEIGHTS))
    parser.add_argument("--ultralytics-root", default=str(DEFAULT_ULTRALYTICS_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fastsam-imgsz", type=int, default=768)
    parser.add_argument("--fastsam-conf", type=float, default=0.10)
    parser.add_argument("--fastsam-iou", type=float, default=0.90)
    parser.add_argument("--min-mask-ratio", type=float, default=0.01)
    parser.add_argument("--max-mask-ratio", type=float, default=0.95)
    parser.add_argument("--max-boxes", type=int, default=0, help="Debug limit. 0 means all boxes.")
    return parser.parse_args()


def resolve_torch_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_ultralytics_device(device: torch.device) -> str | int:
    if device.type == "cuda":
        return 0
    return device.type


def load_fastsam(weights: Path, ultralytics_root: Path):
    if str(ultralytics_root) not in sys.path:
        sys.path.insert(0, str(ultralytics_root))
    from ultralytics import FastSAM

    return FastSAM(str(weights))


def read_bbox_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = dict(raw)
            for key in ("box_index", "class_id", "image_width_px", "image_height_px", "x1", "y1", "x2", "y2"):
                row[key] = int(float(row[key]))
            rows.append(row)
    return rows


def damage_prompt(class_name: str) -> str:
    if class_name in {"D00", "D10", "D20"}:
        return "crack"
    return "road damage"


def draw_bbox(image: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    canvas = image.copy()
    class_name = str(row["class_name"])
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    label = f"{class_name} box{row['box_index']}"
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(canvas, (x1, max(0, y1 - th - 9)), (x1 + tw + 10, max(th + 9, y1)), color, -1)
    cv2.putText(canvas, label, (x1 + 5, max(th + 1, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return canvas


def crop_from_row(image: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    return image[y1:y2, x1:x2].copy()


def masks_to_arrays(masks, crop_shape_hw: tuple[int, int]) -> list[np.ndarray]:
    height, width = crop_shape_hw
    if masks is None or len(masks) == 0:
        return []
    masks_np = masks.detach().float().cpu().numpy()
    arrays: list[np.ndarray] = []
    for item in masks_np:
        mask = item > 0
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        arrays.append(mask.astype(np.uint8))
    return arrays


def select_best_crop_mask(
    masks: list[np.ndarray],
    crop_shape_hw: tuple[int, int],
    min_ratio: float = 0.01,
    max_ratio: float = 0.90,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    crop_pixels = int(crop_shape_hw[0] * crop_shape_hw[1])
    candidates = []
    invalid_reasons = []
    for index, mask in enumerate(masks):
        if mask.shape != crop_shape_hw:
            mask = cv2.resize(mask.astype(np.uint8), (crop_shape_hw[1], crop_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
        pixels = int(mask.sum())
        ratio = pixels / max(1, crop_pixels)
        if pixels < 20:
            invalid_reasons.append(f"mask{index}:too_small:{ratio:.4f}")
            continue
        if ratio < min_ratio:
            invalid_reasons.append(f"mask{index}:ratio_too_small:{ratio:.4f}")
            continue
        if ratio > max_ratio:
            invalid_reasons.append(f"mask{index}:ratio_too_large:{ratio:.4f}")
            continue
        candidates.append((pixels, index, ratio, mask))

    if not candidates:
        return None, {
            "status": "failed",
            "selected_index": -1,
            "selected_mask_pixels": 0,
            "selected_mask_ratio": 0.0,
            "failure_reason": "no_valid_mask;" + "|".join(invalid_reasons[:6]),
        }

    pixels, index, ratio, mask = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return mask, {
        "status": "success",
        "selected_index": int(index),
        "selected_mask_pixels": int(pixels),
        "selected_mask_ratio": float(ratio),
        "failure_reason": "",
    }


def paste_mask_into_image(full_shape_hw: tuple[int, int], crop_mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    full = np.zeros(full_shape_hw, dtype=np.uint8)
    resized = crop_mask
    target_shape = (y2 - y1, x2 - x1)
    if resized.shape != target_shape:
        resized = cv2.resize(resized.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    full[y1:y2, x1:x2] = (resized > 0).astype(np.uint8)
    return full


def fastsam_predict_crop_masks(
    model,
    crop_bgr: np.ndarray,
    crop_shape_hw: tuple[int, int],
    prompt: str,
    args: argparse.Namespace,
    device_name,
):
    prompt_fallback = False
    kwargs = dict(
        source=crop_bgr,
        texts=[prompt],
        device=device_name,
        retina_masks=False,
        imgsz=args.fastsam_imgsz,
        conf=args.fastsam_conf,
        iou=args.fastsam_iou,
        verbose=False,
    )
    start = time.perf_counter()
    try:
        results = model.predict(**kwargs)
    except ModuleNotFoundError as exc:
        if "clip" not in str(exc).lower():
            raise
        prompt_fallback = True
        kwargs.pop("texts", None)
        results = model.predict(**kwargs)
    except RuntimeError:
        kwargs["device"] = "cpu"
        try:
            results = model.predict(**kwargs)
        except ModuleNotFoundError as exc:
            if "clip" not in str(exc).lower():
                raise
            prompt_fallback = True
            kwargs.pop("texts", None)
            results = model.predict(**kwargs)
    elapsed = time.perf_counter() - start
    masks = None if not results or results[0].masks is None else results[0].masks.data
    return masks_to_arrays(masks, crop_shape_hw), elapsed, {"prompt_fallback_to_segment_all": prompt_fallback}


def render_all_masks(crop: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    vis = np.zeros_like(crop)
    for index, mask in enumerate(masks):
        color = FASTSAM_COLORS[index % len(FASTSAM_COLORS)]
        vis[mask > 0] = color
    return overlay_mask(crop, vis, alpha=0.45)


def overlay_mask(image: np.ndarray, mask_or_vis: np.ndarray, alpha: float = 0.42, color=(0, 255, 0)) -> np.ndarray:
    if mask_or_vis.ndim == 2:
        vis = np.zeros_like(image)
        vis[mask_or_vis > 0] = color
    else:
        vis = mask_or_vis
    return cv2.addWeighted(image, 1.0 - alpha, vis, alpha, 0)


def add_caption(image: np.ndarray, caption: str) -> np.ndarray:
    bar = np.full((42, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, caption, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([image, bar])


def fit_panel(image: np.ndarray, panel_w: int = 360, panel_h: int = 360) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(panel_w / max(1, width), panel_h / max(1, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_board(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    rendered = [add_caption(fit_panel(image), title) for title, image in panels]
    separator = np.full((rendered[0].shape[0], 10, 3), 245, dtype=np.uint8)
    canvas = rendered[0]
    for image in rendered[1:]:
        canvas = np.hstack([canvas, separator, image])
    return canvas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root).expanduser().resolve()
    bbox_csv = Path(args.bbox_csv).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    ultralytics_root = Path(args.ultralytics_root).expanduser().resolve()
    fastsam_weights = Path(args.fastsam_weights).expanduser().resolve()
    device = resolve_torch_device(args.device)
    fastsam_device = resolve_ultralytics_device(device)
    if importlib.util.find_spec("clip") is None:
        raise ModuleNotFoundError(
            "FastSAM text prompting requires the `clip` package. "
            "The packaged diagnostic images already exist under assets/area_segmentation_quality_crop/. "
            "Install the Ultralytics CLIP dependency before regenerating these visuals."
        )

    output_dirs = {
        "original_bbox": output_root / "01_original_bbox",
        "crop_raw": output_root / "02_crop_raw",
        "fastsam_prompt_all": output_root / "03_crop_fastsam_prompt_all",
        "fastsam_success": output_root / "04_crop_fastsam_prompt_success",
        "fastsam_failed": output_root / "05_crop_fastsam_prompt_failed",
        "boards": output_root / "boards",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    fastsam = load_fastsam(fastsam_weights, ultralytics_root)
    bbox_rows = read_bbox_rows(bbox_csv)
    if args.max_boxes:
        bbox_rows = bbox_rows[: args.max_boxes]

    rows: list[dict[str, Any]] = []
    for row in bbox_rows:
        image_path = image_root / str(row["image_name"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        crop = crop_from_row(image, row)
        if crop.size == 0:
            continue

        stem = f"{Path(row['image_name']).stem}_box{row['box_index']}_{row['class_name']}"
        prompt = damage_prompt(str(row["class_name"]))

        original_bbox = draw_bbox(image, row)
        crop_shape_hw = crop.shape[:2]
        fastsam_masks, fastsam_time, prompt_meta = fastsam_predict_crop_masks(fastsam, crop, crop_shape_hw, prompt, args, fastsam_device)
        selected_mask, fastsam_meta = select_best_crop_mask(
            fastsam_masks,
            crop_shape_hw,
            min_ratio=args.min_mask_ratio,
            max_ratio=args.max_mask_ratio,
        )

        all_masks_overlay = render_all_masks(crop, fastsam_masks)
        if selected_mask is None:
            selected_overlay_crop = crop.copy()
            cv2.putText(
                selected_overlay_crop,
                "FastSAM crop mask failed",
                (8, max(24, min(crop.shape[0] - 8, 24))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            selected_full = original_bbox.copy()
            success_dir = output_dirs["fastsam_failed"]
        else:
            selected_overlay_crop = overlay_mask(crop, selected_mask, color=(0, 255, 255))
            full_mask = paste_mask_into_image(
                image.shape[:2],
                selected_mask,
                (row["x1"], row["y1"], row["x2"], row["y2"]),
            )
            selected_full = draw_bbox(overlay_mask(image, full_mask, color=(0, 255, 255)), row)
            success_dir = output_dirs["fastsam_success"]

        cv2.imwrite(str(output_dirs["original_bbox"] / f"{stem}.jpg"), original_bbox)
        cv2.imwrite(str(output_dirs["crop_raw"] / f"{stem}.jpg"), crop)
        cv2.imwrite(str(output_dirs["fastsam_prompt_all"] / f"{stem}.jpg"), all_masks_overlay)
        cv2.imwrite(str(success_dir / f"{stem}.jpg"), selected_full)

        board = make_board(
            [
                ("Original + GT bbox", original_bbox),
                ("BBox crop", crop),
                (f"FastSAM prompt: {prompt}", all_masks_overlay),
                (f"Selected mask: {fastsam_meta['status']}", selected_overlay_crop),
            ]
        )
        cv2.imwrite(str(output_dirs["boards"] / f"{stem}.jpg"), board)

        rows.append(
            {
                "image_name": row["image_name"],
                "box_index": row["box_index"],
                "class_name": row["class_name"],
                "crop_width_px": crop.shape[1],
                "crop_height_px": crop.shape[0],
                "fastsam_prompt": prompt,
                "fastsam_prompt_fallback_to_segment_all": prompt_meta["prompt_fallback_to_segment_all"],
                "fastsam_status": fastsam_meta["status"],
                "fastsam_mask_count": len(fastsam_masks),
                "fastsam_selected_index": fastsam_meta["selected_index"],
                "fastsam_selected_mask_pixels": fastsam_meta["selected_mask_pixels"],
                "fastsam_selected_mask_ratio": round(float(fastsam_meta["selected_mask_ratio"]), 6),
                "fastsam_failure_reason": fastsam_meta["failure_reason"],
                "fastsam_time_s": round(fastsam_time, 6),
                "note": "No area is computed; these are crop segmentation quality diagnostics.",
            }
        )

    write_csv(output_root / "crop_segmentation_quality.csv", rows)
    summary = {
        "input_mode": "GT bbox crop only",
        "num_bboxes": len(rows),
        "device": str(device),
        "fastsam_device": str(fastsam_device),
        "fastsam_weight": str(fastsam_weights),
        "fastsam_params": {
            "imgsz": args.fastsam_imgsz,
            "conf": args.fastsam_conf,
            "iou": args.fastsam_iou,
            "min_mask_ratio": args.min_mask_ratio,
            "max_mask_ratio": args.max_mask_ratio,
        },
        "prompt_mapping": {
            "D00": damage_prompt("D00"),
            "D10": damage_prompt("D10"),
            "D20": damage_prompt("D20"),
            "D40": damage_prompt("D40"),
        },
        "notes": [
            "FastSAM is run on bbox crops only; the full image is not passed to FastSAM.",
            "The text prompt is derived from the detected RDD damage class.",
            "PIDNet is not used in this visualization.",
            "Visuals do not include square-meter area values.",
            "Success means the selected crop mask is non-empty and does not nearly fill the crop.",
        ],
        "outputs": {key: str(path) for key, path in output_dirs.items()},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "summary": str(output_root / "summary.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
