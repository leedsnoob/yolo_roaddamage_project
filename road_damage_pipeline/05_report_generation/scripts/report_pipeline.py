#!/usr/bin/env python3
"""Build image/video report evidence and optionally call SiliconFlow Qwen.

The report module is intentionally evidence-grounded. It never feeds RDD GT
boxes to the VLM. Demo detections are model predictions, and every generated
number is written to report_input.json before it is sent to Qwen.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

cv2 = None
requests = None


def require_cv2():
    """Load OpenCV only for runtime paths that need image/video processing."""
    global cv2
    if cv2 is None:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency: opencv-python. Install pipeline requirements before running report demos."
            ) from exc
        cv2 = cv2_module
    return cv2


def require_requests():
    """Load requests only when the SiliconFlow API is actually called."""
    global requests
    if requests is None:
        try:
            import requests as requests_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency: requests. Install pipeline requirements before calling SiliconFlow."
            ) from exc
        requests = requests_module
    return requests


def resolve_roots() -> tuple[Path, Path, Path, Path]:
    """Resolve paths when the pipeline is run from root or inside final YOLO repo."""
    module_root = Path(__file__).resolve().parents[1]
    pipeline_root = module_root.parent

    # GitHub layout: <workspace>/ultralytics_yolo11_final/road_damage_pipeline/...
    if pipeline_root.parent.name == "ultralytics_yolo11_final" and (
        pipeline_root.parent / "ultralytics"
    ).exists():
        repo_root = pipeline_root.parent
        workspace_root = repo_root.parent
    else:
        # Development layout: <workspace>/road_damage_pipeline/...
        workspace_root = pipeline_root.parent
        repo_root = workspace_root / "ultralytics_yolo11_final"

    return module_root, pipeline_root, workspace_root, repo_root


MODULE_ROOT, PIPELINE_ROOT, WORKSPACE_ROOT, DEFAULT_REPO_ROOT = resolve_roots()

DETECTION_ROOT = PIPELINE_ROOT / "02_detection"
VIDEO_ROOT = PIPELINE_ROOT / "03_video_dedup"
AREA_ROOT = PIPELINE_ROOT / "04_area_measurement"
AREA_SCRIPT_ROOT = AREA_ROOT / "scripts"
if str(AREA_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(AREA_SCRIPT_ROOT))
from live_area_engine import BBoxSpec, LiveAreaEngine, build_config_from_args  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_OUTPUT_ROOT = MODULE_ROOT / "outputs"
DEFAULT_IMAGE_DIR = DETECTION_ROOT / "samples" / "images" / "raw"
DEFAULT_VIDEO = VIDEO_ROOT / "samples" / "videos" / "3_dense_130_190.mp4"
DEFAULT_WEIGHTS = DETECTION_ROOT / "weights" / "yolo11s_original_nondrone_noempty_best.pt"
DEFAULT_VIDEO_RESULTS = (
    WORKSPACE_ROOT
    / "video_damage_analytics"
    / "outputs"
    / "dedup_case_study_yolo11s"
    / "3_dense_130_190"
    / "bytetrack"
    / "track_only"
)

CLASS_ID_TO_CODE = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_ID_TO_EN = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}
CLASS_NAME_TO_ID = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
    "Longitudinal Crack": 0,
    "Transverse Crack": 1,
    "Alligator Crack": 2,
    "Pothole": 3,
}
CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (0, 255, 0),
    3: (0, 128, 255),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_bbox(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v) for v in json.loads(value)]


def class_id_from_any(class_id: Any, class_name: str = "") -> int:
    if class_id not in {None, ""}:
        return int(class_id)
    if class_name in CLASS_NAME_TO_ID:
        return CLASS_NAME_TO_ID[class_name]
    raise ValueError(f"Cannot resolve class id from class_name={class_name!r}")


def class_code(class_id: int) -> str:
    return CLASS_ID_TO_CODE.get(int(class_id), f"class_{class_id}")


def class_label(class_id: int, class_name: str = "") -> str:
    code = class_code(class_id)
    readable = class_name or CLASS_ID_TO_EN.get(int(class_id), code)
    return f"{code} {readable}"


def draw_box(frame, xyxy: list[float], class_id: int, label: str) -> None:
    cv2_mod = require_cv2()
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    color = CLASS_COLORS.get(int(class_id), (255, 255, 255))
    cv2_mod.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2_mod.getTextSize(label, cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(y1 - 8, th + 4)
    cv2_mod.rectangle(frame, (x1, y_text - th - 5), (x1 + tw + 8, y_text + 3), color, -1)
    cv2_mod.putText(
        frame,
        label,
        (x1 + 4, y_text - 2),
        cv2_mod.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2_mod.LINE_AA,
    )


def write_predicted_review_grid(image_items: list[dict[str, Any]], out_path: Path) -> str:
    """Write a raw-vs-predicted bbox review grid for quick human inspection."""
    cv2_mod = require_cv2()
    pairs = []
    for item in image_items:
        raw = cv2_mod.imread(str(item["raw_image"]))
        pred = cv2_mod.imread(str(item["predicted_image"]))
        if raw is not None and pred is not None:
            pairs.append((item["image_name"], raw, pred))
    if not pairs:
        return ""

    cell_w, cell_h = 360, 270
    rows = []
    for name, raw, pred in pairs:
        raw = cv2_mod.resize(raw, (cell_w, cell_h), interpolation=cv2_mod.INTER_AREA)
        pred = cv2_mod.resize(pred, (cell_w, cell_h), interpolation=cv2_mod.INTER_AREA)
        for img, title in ((raw, f"RAW {name}"), (pred, f"PRED {name}")):
            cv2_mod.rectangle(img, (0, 0), (cell_w, 28), (245, 245, 245), -1)
            cv2_mod.putText(img, title, (8, 20), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2_mod.LINE_AA)
        rows.append(np.hstack([raw, pred]))
    ensure_dir(out_path.parent)
    cv2_mod.imwrite(str(out_path), np.vstack(rows))
    return str(out_path)


def setup_ultralytics(repo_root: Path):
    cache_dir = ensure_dir(MODULE_ROOT / ".ultralytics_cache")
    os.environ["YOLO_CONFIG_DIR"] = str(cache_dir)
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ultralytics import YOLO

    return YOLO


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


def build_live_area_engine(args: argparse.Namespace) -> LiveAreaEngine:
    try:
        return LiveAreaEngine(build_config_from_args(args))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Live depth area engine failed to initialize: {type(exc).__name__}: {exc}") from exc


def run_image_detection(args: argparse.Namespace, out_dir: Path, area_engine: LiveAreaEngine) -> dict[str, Any]:
    cv2_mod = require_cv2()
    YOLO = setup_ultralytics(args.repo_root)
    model = YOLO(str(args.weights))
    device = resolve_device(args.device)
    image_paths = sorted(args.image_dir.glob("*.jpg"))[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {args.image_dir}")

    pred_dir = ensure_dir(out_dir / "predicted_images")
    area_visual_dir = ensure_dir(out_dir / "area_visuals")
    detection_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []
    image_items: list[dict[str, Any]] = []
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
        detections: list[dict[str, Any]] = []
        area_visuals: dict[str, str] = {}
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy_list = result.boxes.xyxy.cpu().tolist()
            conf_list = result.boxes.conf.cpu().tolist()
            cls_list = result.boxes.cls.int().cpu().tolist()
            box_specs: list[BBoxSpec] = []
            pending: list[dict[str, Any]] = []
            for det_idx, (xyxy, conf, cls_id) in enumerate(zip(xyxy_list, conf_list, cls_list)):
                cls_id = int(cls_id)
                bbox = [float(v) for v in xyxy]
                detection_id = f"{image_path.stem}-det-{det_idx}"
                box_specs.append(
                    BBoxSpec(
                        item_id=detection_id,
                        class_id=cls_id,
                        class_name=CLASS_ID_TO_EN.get(cls_id, str(cls_id)),
                        confidence=float(conf),
                        bbox_xyxy=bbox,
                    )
                )
                pending.append({"detection_id": detection_id, "xyxy": bbox, "conf": float(conf), "cls_id": cls_id})

            area_by_id, area_visuals = area_engine.estimate_image_with_visuals(image_path, box_specs, area_visual_dir)
            for det in pending:
                cls_id = det["cls_id"]
                areas = area_by_id[det["detection_id"]]
                item = {
                    "detection_id": det["detection_id"],
                    "image_name": image_path.name,
                    "class_id": cls_id,
                    "class_code": class_code(cls_id),
                    "class_name": CLASS_ID_TO_EN.get(cls_id, str(cls_id)),
                    "confidence": round(det["conf"], 4),
                    "bbox_xyxy": [round(float(v), 2) for v in det["xyxy"]],
                    "area_estimates": areas,
                }
                detections.append(item)
                detection_rows.append(
                    {
                        "image_name": image_path.name,
                        "detection_id": item["detection_id"],
                        "class_code": item["class_code"],
                        "class_name": item["class_name"],
                        "confidence": item["confidence"],
                        "bbox_xyxy": json.dumps(item["bbox_xyxy"]),
                        "M1_area_m2": areas[0]["estimated_area_m2"],
                        "M3_area_m2": areas[1]["estimated_area_m2"],
                        "M4_area_m2": areas[2]["estimated_area_m2"],
                        "M3_status": areas[1]["status"],
                        "M4_status": areas[2]["status"],
                    }
                )
                for area in areas:
                    area_rows.append(
                        {
                            "image_name": image_path.name,
                            "detection_id": item["detection_id"],
                            "class_code": item["class_code"],
                            "class_name": item["class_name"],
                            "confidence": item["confidence"],
                            "bbox_xyxy": json.dumps(item["bbox_xyxy"]),
                            "method_id": area["method_id"],
                            "method_name": area["method_name"],
                            "estimated_area_m2": area["estimated_area_m2"],
                            "status": area["status"],
                            "mask_source": area["mask_source"],
                            "mask_pixels": area["mask_pixels"],
                            "depth_median_m": area.get("depth_median_m", ""),
                            "raw_depth_bbox_area_m2": area.get("raw_depth_bbox_area_m2", ""),
                            "depth_area_is_assumption": area["depth_area_is_assumption"],
                        }
                    )
                draw_box(
                    image,
                    det["xyxy"],
                    cls_id,
                    f"{class_code(cls_id)} {det['conf']:.2f}",
                )
        pred_path = pred_dir / f"{image_path.stem}_pred.jpg"
        cv2_mod.imwrite(str(pred_path), image)
        image_items.append(
            {
                "image_name": image_path.name,
                "raw_image": str(image_path),
                "predicted_image": str(pred_path),
                "area_visuals": area_visuals if detections else {},
                "num_detections": len(detections),
                "detections": detections,
            }
        )

    write_csv(out_dir / "detections.csv", detection_rows)
    write_csv(out_dir / "area_estimates.csv", area_rows)
    review_grid_path = write_predicted_review_grid(image_items, out_dir / "predicted_review_grid.jpg")
    counts = Counter()
    for row in detection_rows:
        counts[row["class_code"]] += 1
    return {
        "report_type": "image_set",
        "source": {
            "type": "image_set",
            "path": str(args.image_dir),
            "num_images": len(image_items),
        },
        "model": {
            "detector_name": "YOLO11s original_nondrone_noempty demo weight",
            "weights_path": str(args.weights),
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "imgsz": args.imgsz,
            "device_actual": device,
        },
        "area_estimation": {
            "methods": ["M1_empirical_bbox", "M3_depth_anything_v2_empirical_bbox", "M4_metric3d_empirical_bbox"],
            "mode": "live_depth",
            "depth_anything_checkpoint": str(area_engine.config.depth_checkpoint),
            "depth_anything_repo": str(area_engine.config.depth_repo),
            "metric3d_model": area_engine.config.metric3d_model,
        },
        "damage_summary": {
            "total_detections": len(detection_rows),
            "counts_by_class": dict(counts),
        },
        "images": image_items,
        "visual_evidence": [
            {"path": item["predicted_image"], "caption": f"Model-predicted bbox visualization for {item['image_name']}"}
            for item in image_items
        ],
        "source_files": {
            "detections_csv": str(out_dir / "detections.csv"),
            "area_estimates_csv": str(out_dir / "area_estimates.csv"),
            "area_visuals_dir": str(area_visual_dir),
            "predicted_review_grid": review_grid_path,
        },
        "limitations": standard_limitations(),
    }


def run_or_reuse_video_results(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.video_results_dir.exists() and (args.video_results_dir / "summary.json").exists():
        return args.video_results_dir
    command = [
        sys.executable,
        str(VIDEO_ROOT / "scripts" / "infer_video.py"),
        "--video",
        str(args.video),
        "--weights",
        str(args.weights),
        "--repo-root",
        str(args.repo_root),
        "--output-root",
        str(out_dir / "video_inference"),
        "--tracker-backend",
        args.tracker_backend,
        "--dedup-mode",
        "track_only",
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--device",
        args.device,
        "--save-video",
    ]
    subprocess.run(command, check=True)
    return out_dir / "video_inference" / args.video.stem / args.tracker_backend / "track_only"


def select_representative_frames(detection_rows: list[dict[str, str]], top_k: int) -> list[int]:
    grouped = Counter(int(row["frame_idx"]) for row in detection_rows)
    ordered = sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    selected: list[int] = []
    min_gap = 60
    for frame_idx, _ in ordered:
        if all(abs(frame_idx - existing) >= min_gap for existing in selected):
            selected.append(frame_idx)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for frame_idx, _ in ordered:
            if frame_idx not in selected:
                selected.append(frame_idx)
            if len(selected) >= top_k:
                break
    return selected


def save_frame_with_detections(
    video_path: Path,
    frame_idx: int,
    frame_rows: list[dict[str, str]],
    out_path: Path,
    fps: float,
) -> dict[str, Any]:
    cv2_mod = require_cv2()
    cap = cv2_mod.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2_mod.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to extract frame {frame_idx} from {video_path}")

    counts = Counter()
    for row in frame_rows:
        cls_id = int(row["class_id"])
        counts[class_code(cls_id)] += 1
        xyxy = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        label = f"{class_code(cls_id)} {float(row['confidence']):.2f} id={row['track_id']}"
        draw_box(frame, xyxy, cls_id, label)
    cv2_mod.putText(
        frame,
        f"frame={frame_idx} time={frame_idx / fps:.2f}s detections={len(frame_rows)}",
        (18, 36),
        cv2_mod.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2_mod.LINE_AA,
    )
    ensure_dir(out_path.parent)
    cv2_mod.imwrite(str(out_path), frame)
    return {
        "frame_idx": frame_idx,
        "timestamp_s": round(frame_idx / fps, 3),
        "num_detections": len(frame_rows),
        "counts_by_class": dict(counts),
        "predicted_frame": str(out_path),
    }


def extract_raw_frame(video_path: Path, frame_idx: int, out_path: Path) -> Path:
    cv2_mod = require_cv2()
    cap = cv2_mod.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2_mod.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to extract raw frame {frame_idx} from {video_path}")
    ensure_dir(out_path.parent)
    cv2_mod.imwrite(str(out_path), frame)
    return out_path


def select_representative_event_detections(
    event_rows: list[dict[str, str]],
    detection_rows: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, Any]]:
    best_detection: dict[tuple[int, int], dict[str, str]] = {}
    for row in detection_rows:
        key = (int(float(row["track_id"])), int(row["class_id"]))
        current = best_detection.get(key)
        if current is None or float(row["confidence"]) > float(current["confidence"]):
            best_detection[key] = row

    candidates: list[dict[str, Any]] = []
    for row in event_rows:
        cls_id = class_id_from_any(row.get("class_id"), row.get("class_name", ""))
        track_id = int(float(row["track_id"]))
        det = best_detection.get((track_id, cls_id))
        if det is not None:
            bbox = [float(det[key]) for key in ("x1", "y1", "x2", "y2")]
            frame_idx = int(det["frame_idx"])
            confidence = float(det["confidence"])
        else:
            bbox = parse_bbox(row["best_bbox_xyxy"])
            frame_idx = int(row["first_frame"])
            confidence = float(row["max_confidence"])
        candidates.append(
            {
                "event_row": row,
                "class_id": cls_id,
                "track_id": track_id,
                "frame_idx": frame_idx,
                "confidence": confidence,
                "bbox_xyxy": bbox,
            }
        )

    ordered = sorted(candidates, key=lambda item: (-item["confidence"], item["frame_idx"]))
    selected: list[dict[str, Any]] = []
    min_gap = 60
    for item in ordered:
        if all(abs(item["frame_idx"] - existing["frame_idx"]) >= min_gap for existing in selected):
            selected.append(item)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for item in ordered:
            if item not in selected:
                selected.append(item)
            if len(selected) >= top_k:
                break
    return selected


def build_video_report_input(args: argparse.Namespace, out_dir: Path, area_engine: LiveAreaEngine) -> dict[str, Any]:
    result_dir = run_or_reuse_video_results(args, out_dir)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    detection_rows = read_csv(result_dir / "detections.csv")
    event_rows = read_csv(result_dir / "track_events.csv")
    fps = float(summary.get("fps") or 25.0)

    frame_dir = ensure_dir(out_dir / "representative_frames")
    raw_frame_dir = ensure_dir(out_dir / "representative_raw_frames")
    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in detection_rows:
        rows_by_frame[int(row["frame_idx"])].append(row)
    selected_event_dets = select_representative_event_detections(event_rows, detection_rows, args.representative_frames)

    representative_frames = []
    representative_area_by_event: dict[str, list[dict[str, Any]]] = {}
    for index, selected in enumerate(selected_event_dets):
        event_row = selected["event_row"]
        frame_idx = int(selected["frame_idx"])
        raw_frame_path = extract_raw_frame(args.video, frame_idx, raw_frame_dir / f"event_{index + 1:02d}_frame_{frame_idx:06d}.jpg")
        annotated_frame = save_frame_with_detections(
            args.video,
            frame_idx,
            rows_by_frame[frame_idx],
            frame_dir / f"event_{index + 1:02d}_frame_{frame_idx:06d}.jpg",
            fps,
        )
        item_id = str(event_row["event_id"])
        spec = BBoxSpec(
            item_id=item_id,
            class_id=int(selected["class_id"]),
            class_name=str(event_row["class_name"]),
            confidence=float(selected["confidence"]),
            bbox_xyxy=[float(v) for v in selected["bbox_xyxy"]],
        )
        representative_area_by_event[item_id] = area_engine.estimate_image(raw_frame_path, [spec])[item_id]
        annotated_frame.update(
            {
                "representative_event_id": item_id,
                "representative_track_id": int(selected["track_id"]),
                "representative_confidence": round(float(selected["confidence"]), 4),
                "representative_bbox_xyxy": [round(float(v), 2) for v in selected["bbox_xyxy"]],
                "raw_frame": str(raw_frame_path),
                "area_estimates": representative_area_by_event[item_id],
            }
        )
        representative_frames.append(annotated_frame)

    events: list[dict[str, Any]] = []
    event_area_rows: list[dict[str, Any]] = []
    for row in event_rows:
        cls_id = class_id_from_any(row.get("class_id"), row.get("class_name", ""))
        bbox = parse_bbox(row["best_bbox_xyxy"])
        areas = representative_area_by_event.get(row["event_id"], [])
        event = {
            "event_id": row["event_id"],
            "track_id": int(row["track_id"]),
            "class_id": cls_id,
            "class_code": class_code(cls_id),
            "class_name": row["class_name"],
            "first_frame": int(row["first_frame"]),
            "first_timestamp_s": float(row["first_timestamp_s"]),
            "last_frame": int(row["last_frame"]),
            "last_timestamp_s": float(row["last_timestamp_s"]),
            "num_hits": int(row["num_hits"]),
            "max_confidence": float(row["max_confidence"]),
            "best_bbox_xyxy": [round(float(v), 2) for v in bbox],
            "area_estimates": areas,
            "area_status": "live_depth_success" if areas else "not_computed_for_non_representative_video_event",
        }
        events.append(event)
        if areas:
            event_area_rows.append(
                {
                    "event_id": event["event_id"],
                    "class_code": event["class_code"],
                    "class_name": event["class_name"],
                    "track_id": event["track_id"],
                    "max_confidence": event["max_confidence"],
                    "M1_area_m2": areas[0]["estimated_area_m2"],
                    "M3_area_m2": areas[1]["estimated_area_m2"],
                    "M4_area_m2": areas[2]["estimated_area_m2"],
                    "M3_status": areas[1]["status"],
                    "M4_status": areas[2]["status"],
                }
            )
    events.sort(key=lambda item: (-item["max_confidence"], item["first_frame"]))
    write_csv(out_dir / "event_area_estimates.csv", event_area_rows)

    return {
        "report_type": "video",
        "source": {
            "type": "video",
            "path": str(args.video),
            "duration_s": float(summary.get("duration_s") or 0.0),
            "fps": fps,
            "width": summary.get("width"),
            "height": summary.get("height"),
            "total_frames": summary.get("total_frames"),
            "processed_frames": summary.get("processed_frames"),
        },
        "model": {
            "detector_name": "YOLO11s original_nondrone_noempty demo weight",
            "weights_path": str(args.weights),
            "tracker_backend": summary.get("tracker_backend", args.tracker_backend),
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "imgsz": args.imgsz,
            "device_actual": summary.get("device_actual", ""),
        },
        "area_estimation": {
            "methods": ["M1_empirical_bbox", "M3_depth_anything_v2_empirical_bbox", "M4_metric3d_empirical_bbox"],
            "mode": "live_depth_representative_events_only",
            "num_events_with_area": len(event_area_rows),
            "depth_anything_checkpoint": str(area_engine.config.depth_checkpoint),
            "depth_anything_repo": str(area_engine.config.depth_repo),
            "metric3d_model": area_engine.config.metric3d_model,
        },
        "damage_summary": {
            "total_detections": int(summary.get("total_detections", len(detection_rows))),
            "unique_events": int(summary.get("unique_track_events", len(event_rows))),
            "counts_by_class": normalize_count_keys(summary.get("detection_counts_by_class", {})),
            "unique_counts_by_class": normalize_count_keys(summary.get("unique_counts_by_class", {})),
        },
        "representative_frames": representative_frames,
        "events": events,
        "visual_evidence": [
            {
                "path": frame["predicted_frame"],
                "caption": f"Predicted bbox keyframe at {frame['timestamp_s']}s with {frame['num_detections']} detections",
            }
            for frame in representative_frames
        ],
        "limitations": standard_limitations(),
        "source_files": {
            "summary_json": str(result_dir / "summary.json"),
            "detections_csv": str(result_dir / "detections.csv"),
            "track_events_csv": str(result_dir / "track_events.csv"),
            "event_area_estimates_csv": str(out_dir / "event_area_estimates.csv"),
        },
    }


def standard_limitations() -> list[str]:
    return [
        "Only model-predicted bounding boxes are used as report evidence; RDD ground-truth boxes are not provided to the VLM.",
        "Confidence is model confidence, not accuracy. Accuracy requires manually labeled evaluation data.",
        "All area values are estimated areas. No camera calibration, lane-line calibration, or physical area ground truth is available in this demo.",
        "M3/M4 are live depth-based estimates over bbox rectangles with assumed FOV; they are not physical ground truth.",
        "Maintenance suggestions are decision support only and are not final engineering diagnosis.",
    ]


def normalize_count_keys(counts: dict[str, Any]) -> dict[str, int]:
    normalized: Counter[str] = Counter()
    for raw_key, value in (counts or {}).items():
        if raw_key in CLASS_NAME_TO_ID:
            key = class_code(CLASS_NAME_TO_ID[raw_key])
        else:
            key = str(raw_key)
        normalized[key] += int(value)
    return dict(normalized)


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def compact_local_paths(value: Any) -> Any:
    """Keep prompts readable by replacing local absolute paths with filenames."""
    if isinstance(value, dict):
        return {key: compact_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_local_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        return path.name or str(path)
    return value


def compact_report_input_for_prompt(report_input: dict[str, Any], max_events: int) -> dict[str, Any]:
    compact = json.loads(json.dumps(report_input, ensure_ascii=False))
    if compact.get("report_type") == "image_set":
        detection_table = []
        class_confidences: dict[str, list[float]] = defaultdict(list)
        image_summary = []
        for image in compact.get("images", []):
            image_counts: Counter[str] = Counter()
            image_top_detections = []
            for det in image.get("detections", []):
                area_by_id = {row["method_id"]: row["estimated_area_m2"] for row in det.get("area_estimates", [])}
                class_confidences[det["class_code"]].append(float(det["confidence"]))
                image_counts[det["class_code"]] += 1
                image_top_detections.append(
                    {
                        "detection_id": det["detection_id"],
                        "class_code": det["class_code"],
                        "confidence": det["confidence"],
                        "M1_area_m2": area_by_id.get("M1"),
                        "M3_area_m2": area_by_id.get("M3"),
                        "M4_area_m2": area_by_id.get("M4"),
                    }
                )
                detection_table.append(
                    {
                        "detection_id": det["detection_id"],
                        "image_name": image["image_name"],
                        "class_code": det["class_code"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "M1_area_m2": area_by_id.get("M1"),
                        "M3_area_m2": area_by_id.get("M3"),
                        "M4_area_m2": area_by_id.get("M4"),
                    }
                )
            image_top_detections.sort(key=lambda row: (-float(row["confidence"]), row["detection_id"]))
            image_summary.append(
                {
                    "image_name": image["image_name"],
                    "num_detections": image["num_detections"],
                    "counts_by_class": dict(sorted(image_counts.items())),
                    "top_detections_by_confidence": image_top_detections[:3],
                }
            )
        compact["image_detection_table"] = detection_table
        compact["image_summary"] = image_summary
        compact["image_class_summary"] = [
            {
                "class_code": class_code,
                "count": len(values),
                "min_confidence": round(min(values), 4),
                "max_confidence": round(max(values), 4),
            }
            for class_code, values in sorted(class_confidences.items())
        ]
        priority_events = sorted(
            detection_table,
            key=lambda row: (-float(row["confidence"]), -float(row.get("M1_area_m2") or 0), row["detection_id"]),
        )[:5]
        compact["priority_events"] = priority_events
    if compact.get("report_type") == "video":
        events = compact.get("events", [])
        compact["events"] = events[:max_events]
        compact["event_list_note"] = (
            f"Only top {min(max_events, len(events))} events by confidence are included in the prompt. "
            f"Full event list with {len(events)} events is stored in report_input.json."
        )
    return compact_local_paths(compact)


def build_qwen_messages(
    report_input: dict[str, Any],
    max_prompt_events: int,
    max_visual_evidence: int,
    report_language: str,
) -> list[dict[str, Any]]:
    compact = compact_report_input_for_prompt(report_input, max_prompt_events)
    if report_language == "en":
        system_prompt = (
            "You are a road-damage inspection report assistant. Write only from the provided structured evidence "
            "and images. Do not invent defects, numbers, areas, accuracy, or engineering conclusions. "
            "Confidence must be described as model confidence, and area must be described as estimated area. "
            "Maintenance recommendations are decision-support suggestions only."
        )
        user_text = (
            "Generate an English Markdown road-damage inspection report from the report_input below. "
            "The report must include: inspection overview, damage statistics, representative-image notes, "
            "priority damage events, M1/M3/M4 estimated-area comparison, maintenance-priority suggestions, and limitations. "
            "Do not output JSON. For damage statistics, prioritize class names and counts. If confidence ranges are used, "
            "copy them only from image_class_summary. Do not recalculate them. Representative-image notes must follow "
            "image_summary.num_detections and image_summary.counts_by_class; do not freely count visible objects from images. "
            "Priority events must use priority_events only and copy each event's detection_id, class_code, confidence, and M1/M3/M4 values. "
            "If a detection_id is referenced, use class_code, confidence, and area values from the same row of image_detection_table. "
            "Do not use the word accuracy unless explicitly saying that confidence is not accuracy.\n\n"
            f"report_input:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )
    else:
        system_prompt = (
            "你是道路病害巡检报告助手。必须只根据用户提供的结构化证据和图片写报告。"
            "不要编造新的病害、数字、面积、准确率或维修结论。confidence 只能称为模型置信度，"
            "area 必须称为估计面积。维护建议只能作为辅助建议。"
        )
        user_text = (
            "请根据下面的 report_input 生成中文 Markdown 道路病害报告。"
            "报告必须包含：巡检概况、病害统计、代表图说明、重点病害事件、M1/M3/M4 面积估计比较、"
            "维护优先级建议、局限性。不要输出 JSON。"
            "病害统计表优先写类别和数量；如果写置信度范围，必须逐字使用 image_class_summary，不要自己重新聚合。"
            "代表图说明必须逐字依据 image_summary 的 num_detections 和 counts_by_class，不要从图片外观自由计数或描述“可见几条”。"
            "重点病害事件必须只使用 priority_events，复制同一行的 detection_id、class_code、confidence、M1/M3/M4，不能自行挑选或混用。"
            "如果引用 detection_id，必须使用 image_detection_table 中同一行的 class_code、confidence 和面积值，"
            "不要把不同 detection 的类别或置信度混用。不要说“准确率”，只能说“模型置信度”。\n\n"
            f"report_input:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for item in report_input.get("visual_evidence", [])[:max_visual_evidence]:
        image_path = Path(item["path"])
        if image_path.exists():
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scrubbed = json.loads(json.dumps(payload, ensure_ascii=False))
    for message in scrubbed.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        prefix = url.split(",", 1)[0]
                        item["image_url"]["url"] = f"{prefix},<base64 omitted>"
    return scrubbed


def call_siliconflow(payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not set. Export it in your shell before using --call-api.")
    requests_mod = require_requests()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests_mod.post(API_URL, headers=headers, json=payload, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"SiliconFlow request failed: {type(exc).__name__}: {exc}") from exc
    if not response.ok:
        raise RuntimeError(f"SiliconFlow HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def write_report_from_response(out_dir: Path, response: dict[str, Any]) -> str:
    content = response["choices"][0]["message"]["content"]
    (out_dir / "report.md").write_text(content, encoding="utf-8")
    return content


def build_payload(args: argparse.Namespace, report_input: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": args.qwen_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": build_qwen_messages(
            report_input,
            args.max_prompt_events,
            args.max_visual_evidence,
            args.report_language,
        ),
    }
    if args.enable_thinking:
        payload["enable_thinking"] = True
        payload["thinking_budget"] = args.thinking_budget
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image/video road-damage reports with SiliconFlow Qwen.")
    parser.add_argument("--mode", choices=["image", "video"], required=True, help="Build an image-set or video report demo.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--video-results-dir", type=Path, default=DEFAULT_VIDEO_RESULTS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker-backend", default="bytetrack", choices=["bytetrack", "botsort", "deepocsort"])
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--representative-frames", type=int, default=3)
    parser.add_argument("--scale-factor", type=float, default=0.01)
    parser.add_argument("--assumed-horizontal-fov-deg", type=float, default=70.0)
    parser.add_argument("--depth-repo", type=Path, default=None)
    parser.add_argument("--depth-checkpoint", type=Path, default=None)
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--metric3d-model", default="metric3d_vit_small")
    parser.add_argument("--metric3d-input-height", type=int, default=616)
    parser.add_argument("--metric3d-input-width", type=int, default=1064)
    parser.add_argument("--max-prompt-events", type=int, default=30)
    parser.add_argument("--max-visual-evidence", type=int, default=5)
    parser.add_argument("--report-language", choices=["zh", "en"], default="zh")
    parser.add_argument("--qwen-model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--call-api", action="store_true", help="Actually call SiliconFlow. Without this flag only local evidence and request preview are written.")
    parser.add_argument("--timeout-s", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    demo_dir = ensure_dir(args.output_root / f"{args.mode}_demo")
    try:
        area_engine = build_live_area_engine(args)
    except RuntimeError as exc:
        (demo_dir / "report.md").write_text(f"Report generation failed before evidence build: {exc}\n", encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.mode == "image":
        report_input = run_image_detection(args, demo_dir, area_engine)
    else:
        report_input = build_video_report_input(args, demo_dir, area_engine)

    report_input["inspection_id"] = f"{args.mode}_demo_{int(start)}"
    report_input["report_generation"] = {
        "provider": "SiliconFlow",
        "model": args.qwen_model,
        "language": args.report_language,
        "api_url": API_URL,
        "call_api": bool(args.call_api),
        "runtime_started_at_unix": int(start),
    }
    write_json(demo_dir / "report_input.json", report_input)

    damage_summary = report_input.get("damage_summary", {})
    if int(damage_summary.get("total_detections", 0) or 0) == 0:
        (demo_dir / "report.md").write_text(
            "No road damage detections were found. Area estimation and Qwen report generation were skipped.\n",
            encoding="utf-8",
        )
        write_json(
            demo_dir / "qwen_request_preview.json",
            {
                "skipped": True,
                "reason": "no_detections",
                "message": "No Qwen request was built because there were no model detections.",
            },
        )
        result = {
            "mode": args.mode,
            "output_dir": str(demo_dir),
            "report_input": str(demo_dir / "report_input.json"),
            "request_preview": str(demo_dir / "qwen_request_preview.json"),
            "report": str(demo_dir / "report.md"),
            "call_api": False,
            "runtime_s": round(time.time() - start, 3),
            "skipped_reason": "no_detections",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    payload = build_payload(args, report_input)
    write_json(demo_dir / "qwen_request_preview.json", preview_payload(payload))

    try:
        if args.call_api:
            response = call_siliconflow(payload, args.timeout_s)
            write_json(demo_dir / "raw_response.json", response)
            write_report_from_response(demo_dir, response)
        else:
            (demo_dir / "report.md").write_text(
                "API call was not executed. Re-run with --call-api after setting SILICONFLOW_API_KEY.\n",
                encoding="utf-8",
            )
    except Exception as exc:  # noqa: BLE001
        (demo_dir / "report.md").write_text(f"Report generation failed: {exc}\n", encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = {
        "mode": args.mode,
        "output_dir": str(demo_dir),
        "report_input": str(demo_dir / "report_input.json"),
        "request_preview": str(demo_dir / "qwen_request_preview.json"),
        "report": str(demo_dir / "report.md"),
        "call_api": bool(args.call_api),
        "runtime_s": round(time.time() - start, 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
