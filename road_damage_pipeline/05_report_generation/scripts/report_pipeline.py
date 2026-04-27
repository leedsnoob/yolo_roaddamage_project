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
from statistics import median
from typing import Any

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
AREA_WIDE_CSV = AREA_ROOT / "assets" / "area" / "four_method_area_results_wide.csv"

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


def empirical_m1_area(class_id: int, width_px: float, height_px: float, scale_factor: float = 0.01) -> tuple[float, float, float]:
    width_m = width_px * scale_factor
    height_m = height_px * scale_factor
    if int(class_id) == 0:
        area_m2 = height_m * 0.8
    elif int(class_id) == 1:
        area_m2 = width_m * 1.2
    elif int(class_id) in {2, 3}:
        area_m2 = width_m * height_m / 3.0
    else:
        area_m2 = width_m * height_m / 3.0
    return width_m, height_m, area_m2


def load_area_ratio_priors(path: Path = AREA_WIDE_CSV) -> dict[str, dict[str, float]]:
    """Use packaged M3/M4 experiment ratios as fast demo priors.

    The full Depth Anything / Metric3D experiments are preserved in module 04.
    For report demos we derive M3/M4 from M1 using per-class median ratios to
    avoid re-running heavy depth models for hundreds of video events.
    """
    rows = read_csv(path)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"m3": [], "m4": []})
    for row in rows:
        cls = row["class_name"]
        for key, column in (("m3", "M3_over_M1"), ("m4", "M4_over_M1")):
            value = row.get(column, "").strip()
            if value:
                grouped[cls][key].append(float(value))

    priors: dict[str, dict[str, float]] = {}
    for cls, values in grouped.items():
        priors[cls] = {
            "m3_over_m1_median": float(median(values["m3"])) if values["m3"] else 1.0,
            "m4_over_m1_median": float(median(values["m4"])) if values["m4"] else 1.0,
        }
    for class_id, code in CLASS_ID_TO_CODE.items():
        priors.setdefault(code, {"m3_over_m1_median": 1.0, "m4_over_m1_median": 1.0})
    return priors


def area_estimates_for_bbox(class_id: int, xyxy: list[float], ratio_priors: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = xyxy
    width_px = max(float(x2) - float(x1), 0.0)
    height_px = max(float(y2) - float(y1), 0.0)
    width_m, height_m, m1 = empirical_m1_area(class_id, width_px, height_px)
    code = class_code(class_id)
    ratios = ratio_priors.get(code, {"m3_over_m1_median": 1.0, "m4_over_m1_median": 1.0})
    m3 = m1 * ratios["m3_over_m1_median"]
    m4 = m1 * ratios["m4_over_m1_median"]
    common = {
        "width_px": round(width_px, 2),
        "height_px": round(height_px, 2),
        "width_m": round(width_m, 4),
        "height_m": round(height_m, 4),
        "scale_assumption": "fixed 0.01 m/px; no camera calibration or lane-line calibration in packaged demo",
    }
    return [
        {
            "method_id": "M1",
            "method_name": "senior empirical bbox rule",
            "estimated_area_m2": round(m1, 6),
            "status": "success",
            "limitation": "bbox-based empirical estimate, not physical GT",
            **common,
        },
        {
            "method_id": "M3",
            "method_name": "Depth Anything V2 bbox-depth empirical ratio",
            "estimated_area_m2": round(m3, 6),
            "status": "packaged_ratio_demo",
            "ratio_to_m1": round(ratios["m3_over_m1_median"], 6),
            "limitation": "demo uses packaged per-class median M3/M1 ratio from module 04; rerun depth module for full per-image depth",
            **common,
        },
        {
            "method_id": "M4",
            "method_name": "Metric3D bbox-depth empirical ratio",
            "estimated_area_m2": round(m4, 6),
            "status": "packaged_ratio_demo",
            "ratio_to_m1": round(ratios["m4_over_m1_median"], 6),
            "limitation": "demo uses packaged per-class median M4/M1 ratio from module 04; rerun Metric3D module for full per-image depth",
            **common,
        },
    ]


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


def run_image_detection(args: argparse.Namespace, out_dir: Path, ratio_priors: dict[str, dict[str, float]]) -> dict[str, Any]:
    cv2_mod = require_cv2()
    YOLO = setup_ultralytics(args.repo_root)
    model = YOLO(str(args.weights))
    device = resolve_device(args.device)
    image_paths = sorted(args.image_dir.glob("*.jpg"))[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {args.image_dir}")

    pred_dir = ensure_dir(out_dir / "predicted_images")
    detection_rows: list[dict[str, Any]] = []
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
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy_list = result.boxes.xyxy.cpu().tolist()
            conf_list = result.boxes.conf.cpu().tolist()
            cls_list = result.boxes.cls.int().cpu().tolist()
            for det_idx, (xyxy, conf, cls_id) in enumerate(zip(xyxy_list, conf_list, cls_list)):
                cls_id = int(cls_id)
                areas = area_estimates_for_bbox(cls_id, [float(v) for v in xyxy], ratio_priors)
                item = {
                    "detection_id": f"{image_path.stem}-det-{det_idx}",
                    "image_name": image_path.name,
                    "class_id": cls_id,
                    "class_code": class_code(cls_id),
                    "class_name": CLASS_ID_TO_EN.get(cls_id, str(cls_id)),
                    "confidence": round(float(conf), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in xyxy],
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
                    }
                )
                draw_box(
                    image,
                    [float(v) for v in xyxy],
                    cls_id,
                    f"{class_code(cls_id)} {float(conf):.2f}",
                )
        pred_path = pred_dir / f"{image_path.stem}_pred.jpg"
        cv2_mod.imwrite(str(pred_path), image)
        image_items.append(
            {
                "image_name": image_path.name,
                "raw_image": str(image_path),
                "predicted_image": str(pred_path),
                "num_detections": len(detections),
                "detections": detections,
            }
        )

    write_csv(out_dir / "detections.csv", detection_rows)
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
        "damage_summary": {
            "total_detections": len(detection_rows),
            "counts_by_class": dict(counts),
        },
        "images": image_items,
        "visual_evidence": [
            {"path": item["predicted_image"], "caption": f"Model-predicted bbox visualization for {item['image_name']}"}
            for item in image_items
        ],
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


def build_video_report_input(args: argparse.Namespace, out_dir: Path, ratio_priors: dict[str, dict[str, float]]) -> dict[str, Any]:
    result_dir = run_or_reuse_video_results(args, out_dir)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    detection_rows = read_csv(result_dir / "detections.csv")
    event_rows = read_csv(result_dir / "track_events.csv")
    fps = float(summary.get("fps") or 25.0)

    frame_dir = ensure_dir(out_dir / "representative_frames")
    selected_frames = select_representative_frames(detection_rows, args.representative_frames)
    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in detection_rows:
        rows_by_frame[int(row["frame_idx"])].append(row)
    representative_frames = [
        save_frame_with_detections(
            args.video,
            frame_idx,
            rows_by_frame[frame_idx],
            frame_dir / f"frame_{frame_idx:06d}.jpg",
            fps,
        )
        for frame_idx in selected_frames
    ]

    events: list[dict[str, Any]] = []
    event_area_rows: list[dict[str, Any]] = []
    for row in event_rows:
        cls_id = class_id_from_any(row.get("class_id"), row.get("class_name", ""))
        bbox = parse_bbox(row["best_bbox_xyxy"])
        areas = area_estimates_for_bbox(cls_id, bbox, ratio_priors)
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
        }
        events.append(event)
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
        "M3/M4 report demo values are derived from packaged area-module depth/Metric3D ratios for speed; rerun module 04 for full per-image depth inference.",
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


def compact_report_input_for_prompt(report_input: dict[str, Any], max_events: int) -> dict[str, Any]:
    compact = json.loads(json.dumps(report_input, ensure_ascii=False))
    if compact.get("report_type") == "video":
        events = compact.get("events", [])
        compact["events"] = events[:max_events]
        compact["event_list_note"] = (
            f"Only top {min(max_events, len(events))} events by confidence are included in the prompt. "
            f"Full event list with {len(events)} events is stored in report_input.json."
        )
    return compact


def build_qwen_messages(report_input: dict[str, Any], max_prompt_events: int) -> list[dict[str, Any]]:
    compact = compact_report_input_for_prompt(report_input, max_prompt_events)
    system_prompt = (
        "你是道路病害巡检报告助手。必须只根据用户提供的结构化证据和图片写报告。"
        "不要编造新的病害、数字、面积、准确率或维修结论。confidence 只能称为模型置信度，"
        "area 必须称为估计面积。维护建议只能作为辅助建议。"
    )
    user_text = (
        "请根据下面的 report_input 生成中文 Markdown 道路病害报告。"
        "报告必须包含：巡检概况、病害统计、代表图说明、重点病害事件、M1/M3/M4 面积估计比较、"
        "维护优先级建议、局限性。不要输出 JSON。\n\n"
        f"report_input:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for item in report_input.get("visual_evidence", [])[:4]:
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
    response = requests_mod.post(API_URL, headers=headers, json=payload, timeout=timeout_s)
    response.raise_for_status()
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
        "messages": build_qwen_messages(report_input, args.max_prompt_events),
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
    parser.add_argument("--max-prompt-events", type=int, default=30)
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
    ratio_priors = load_area_ratio_priors()

    if args.mode == "image":
        report_input = run_image_detection(args, demo_dir, ratio_priors)
    else:
        report_input = build_video_report_input(args, demo_dir, ratio_priors)

    report_input["inspection_id"] = f"{args.mode}_demo_{int(start)}"
    report_input["report_generation"] = {
        "provider": "SiliconFlow",
        "model": args.qwen_model,
        "api_url": API_URL,
        "call_api": bool(args.call_api),
        "runtime_started_at_unix": int(start),
    }
    write_json(demo_dir / "report_input.json", report_input)

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
    except RuntimeError as exc:
        (demo_dir / "report.md").write_text(f"Report generation failed before API call: {exc}\n", encoding="utf-8")
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
