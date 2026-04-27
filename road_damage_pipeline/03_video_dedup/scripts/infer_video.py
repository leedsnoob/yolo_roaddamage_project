import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_core.dedup_utils import (
    compute_event_metrics,
    load_homography_config,
    merge_events_by_plane,
    select_default_tracker,
    write_csv,
)
from pipeline_core.tracker_backends import DeepOCSortAdapter, tracker_backend_to_yaml


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_VIDEO = PIPELINE_ROOT / "samples" / "videos" / "3_dense_130_190.mp4"
DEFAULT_WEIGHTS = PIPELINE_ROOT / "weights" / "yolo11s_original_nondrone_noempty_best.pt"
DEFAULT_REPO_ROOT = WORKSPACE_ROOT / "ultralytics_yolo11_final"
DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "infer_video_default.yaml"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "outputs"

CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (0, 255, 0),
    3: (0, 128, 255),
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDD road-damage video inference with tracker comparison and dedup.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--video", type=Path, help="Input video path.")
    parser.add_argument("--weights", type=Path, help="Model weights path.")
    parser.add_argument("--repo-root", type=Path, help="Ultralytics repo root matching the trained model.")
    parser.add_argument("--output-root", type=Path, help="Directory for outputs.")
    parser.add_argument("--device", type=str, help="Device: auto|mps|cpu.")
    parser.add_argument("--imgsz", type=int, help="Inference image size.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold.")
    parser.add_argument("--tracker", type=str, help="Tracker yaml name or absolute path for Ultralytics trackers.")
    parser.add_argument("--tracker-backend", type=str, help="Tracker backend: bytetrack|botsort|deepocsort.")
    parser.add_argument("--dedup-mode", type=str, help="Dedup mode: track_only|track_plus_plane.")
    parser.add_argument("--homography-config", type=Path, help="Optional YAML homography config for plane merge.")
    parser.add_argument("--compare-trackers", action="store_true", help="Run ByteTrack, BoT-SORT and DeepOC-SORT comparison.")
    parser.add_argument("--stride", type=int, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, help="Process at most this many sampled frames.")
    parser.add_argument("--warmup-frames", type=int, help="Extra warmup runs before actual processing. Used mainly for MPS.")
    parser.add_argument("--render-mode", type=str, help="Video render mode: analysis or preview.")
    parser.add_argument("--render-hold-frames", type=int, help="Keep drawing the last bbox for this many missed frames.")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video.")
    parser.add_argument("--disable-fallback", action="store_true", help="Unused legacy flag kept for compatibility.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    config = load_yaml(args.config)
    merged = {
        "video": DEFAULT_VIDEO,
        "weights": DEFAULT_WEIGHTS,
        "repo_root": DEFAULT_REPO_ROOT,
        "output_root": DEFAULT_OUTPUT_ROOT,
        "device": "auto",
        "imgsz": 832,
        "conf": 0.25,
        "iou": 0.5,
        "tracker": "bytetrack.yaml",
        "tracker_backend": "bytetrack",
        "dedup_mode": "track_only",
        "homography_config": None,
        "compare_trackers": False,
        "stride": 1,
        "max_frames": 0,
        "save_video": False,
        "render_mode": "analysis",
        "track_iou_thresh": 0.3,
        "track_max_gap": 8,
        "warmup_frames": 2,
        "render_hold_frames": 0,
        "fragmentation_gap_frames": 15,
        "fragmentation_iou_threshold": 0.5,
        "deepocsort_max_age": 30,
        "deepocsort_min_hits": 1,
        "deepocsort_delta_t": 3,
        "deepocsort_inertia": 0.2,
    }
    merged.update(config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    merged["save_video"] = bool(args.save_video or merged.get("save_video"))
    merged["compare_trackers"] = bool(args.compare_trackers or merged.get("compare_trackers"))
    for key in ("video", "weights", "repo_root", "output_root"):
        merged[key] = Path(merged[key])
    if merged.get("homography_config"):
        merged["homography_config"] = Path(merged["homography_config"])
    return merged


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sw_vers() -> str:
    try:
        result = subprocess.run(["sw_vers"], capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception as exc:
        return f"sw_vers failed: {exc}"


def probe_mps() -> dict:
    status = {
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "sw_vers": sw_vers(),
    }
    try:
        _ = torch.ones(2, device="mps")
        status["mps_tensor_test"] = "ok"
    except Exception as exc:
        status["mps_tensor_test"] = repr(exc)
    return status


def resolve_device(device_arg: str) -> tuple[str, dict]:
    mps_status = probe_mps()
    mps_ready = bool(mps_status["mps_available"]) and mps_status["mps_tensor_test"] == "ok"
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "0", mps_status
        if mps_ready:
            return "mps", mps_status
        return "cpu", mps_status
    if device_arg == "mps":
        if not mps_ready:
            raise RuntimeError(
                "MPS was requested explicitly but is not usable in the current environment. "
                f"Probe result: {json.dumps(mps_status, ensure_ascii=False)}"
            )
        return "mps", mps_status
    return device_arg, mps_status


def warmup_model(model, frame, cfg: dict, device: str) -> dict:
    warmup_frames = max(int(cfg.get("warmup_frames", 0)), 0)
    info = {
        "enabled": bool(device == "mps" and warmup_frames > 0),
        "warmup_frames": warmup_frames,
        "warmup_runtime_s": 0.0,
    }
    if not info["enabled"]:
        return info
    start = time.time()
    for _ in range(warmup_frames):
        model.predict(
            frame,
            conf=float(cfg["conf"]),
            iou=float(cfg["iou"]),
            imgsz=int(cfg["imgsz"]),
            device=device,
            verbose=False,
        )[0]
    info["warmup_runtime_s"] = round(time.time() - start, 3)
    return info


def setup_ultralytics(repo_root: Path):
    cache_dir = ensure_dir(PIPELINE_ROOT / ".ultralytics_cache")
    os.environ["YOLO_CONFIG_DIR"] = str(cache_dir)
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ultralytics import YOLO

    return YOLO


def require_module(module_name: str, reason: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Missing Python module `{module_name}` required for {reason}. "
            "Install dependencies with the workspace requirements used by the pipeline."
        )


def check_runtime_dependencies(tracker_backend: str) -> None:
    if tracker_backend in {"bytetrack", "botsort"}:
        require_module("scipy", f"Ultralytics {tracker_backend} tracker")
    if tracker_backend == "deepocsort":
        require_module("filterpy", "DeepOC-SORT-lite Kalman filter")


def draw_box(frame, xyxy, color, label):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(y1 - 8, th + 4)
    cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 6, y_text + 2), color, -1)
    cv2.putText(frame, label, (x1 + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def output_video_fps(source_fps: float, stride: int, render_mode: str) -> float:
    if render_mode == "preview":
        return source_fps
    return max(source_fps / max(int(stride), 1), 1.0)


def render_mode_writes_non_inference_frames(render_mode: str) -> bool:
    return render_mode == "preview"


def stale_render_items(render_cache: dict, current_keys: set[str], frame_idx: int, hold_frames: int) -> list[dict]:
    if hold_frames <= 0:
        render_cache.clear()
        return []

    stale_items = []
    expired_keys = []
    for event_key, item in render_cache.items():
        if event_key in current_keys:
            continue
        if frame_idx - item["last_seen_frame"] <= hold_frames:
            stale_items.append(item)
        else:
            expired_keys.append(event_key)
    for event_key in expired_keys:
        render_cache.pop(event_key, None)
    return stale_items


def tracker_config_path(repo_root: Path, tracker_name: str) -> Path:
    tracker_path = Path(tracker_name)
    if tracker_path.is_file():
        return tracker_path
    return repo_root / "ultralytics" / "cfg" / "trackers" / tracker_name


def build_output_dir(cfg: dict) -> Path:
    return ensure_dir(cfg["output_root"] / cfg["video"].stem / str(cfg["tracker_backend"]) / str(cfg["dedup_mode"]))


def extract_predict_detections(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy_list = result.boxes.xyxy.cpu().tolist()
    conf_list = result.boxes.conf.cpu().tolist()
    cls_list = result.boxes.cls.int().cpu().tolist()
    return [
        {
            "xyxy": [float(v) for v in xyxy],
            "confidence": float(conf),
            "class_id": int(cls_id),
        }
        for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list)
    ]


def build_event_rows(events):
    rows = []
    for event in events.values():
        best_bbox = event["best_bbox_xyxy"]
        rows.append(
            {
                "event_id": event["event_id"],
                "class_id": event["class_id"],
                "class_name": event["class_name"],
                "track_id": event["track_id"],
                "first_frame": event["first_frame"],
                "first_timestamp_s": round(event["first_timestamp_s"], 3),
                "last_frame": event["last_frame"],
                "last_timestamp_s": round(event["last_timestamp_s"], 3),
                "duration_s": round(event["last_timestamp_s"] - event["first_timestamp_s"], 3),
                "num_hits": event["num_hits"],
                "max_confidence": round(event["max_confidence"], 4),
                "best_bbox_xyxy": json.dumps([round(v, 2) for v in best_bbox], ensure_ascii=False),
            }
        )
    rows.sort(key=lambda row: (row["first_frame"], row["class_id"], row["track_id"]))
    return rows


def summary_note_block():
    return {
        "dedup_strategy": "track_only counts one class-specific track_id as one event. track_plus_plane additionally merges short-gap spatially consistent events on a configured road plane.",
        "accuracy": "Confidence is not accuracy. True precision/recall/F1 require a labeled video benchmark.",
        "wiou": "WIoU is only the training-time bounding box regression loss. It is not the video deduplication algorithm.",
        "damageqwen": "The referenced 2025 DamageQwen paper removes duplicate detections inside a single image using CLIP similarity. It does not solve cross-frame deduplication.",
        "synthetic_ids": "If an Ultralytics tracker returns detections without track ids, those detections are treated as unassociated and receive one-off synthetic ids so the fragmentation is not hidden.",
    }


def run_single(cfg: dict) -> dict:
    for path_key in ("video", "weights", "repo_root"):
        if not cfg[path_key].exists():
            raise FileNotFoundError(f"{path_key} not found: {cfg[path_key]}")
    if cfg["dedup_mode"] == "track_plus_plane" and not cfg.get("homography_config"):
        raise ValueError("track_plus_plane requires --homography-config")

    YOLO = setup_ultralytics(cfg["repo_root"])
    device, mps_status = resolve_device(cfg["device"])
    model = YOLO(str(cfg["weights"]))
    names = model.names

    tracker_backend = str(cfg["tracker_backend"])
    check_runtime_dependencies(tracker_backend)
    tracker_yaml_name = None if tracker_backend == "deepocsort" else (cfg.get("tracker") or tracker_backend_to_yaml(tracker_backend))
    tracker_cfg_path = tracker_config_path(cfg["repo_root"], tracker_yaml_name) if tracker_yaml_name else None
    deepocsort = None
    if tracker_backend == "deepocsort":
        deepocsort = DeepOCSortAdapter(
            det_thresh=float(cfg["conf"]),
            iou_threshold=float(cfg.get("track_iou_thresh", 0.3)),
            max_age=int(cfg.get("deepocsort_max_age", 30)),
            min_hits=int(cfg.get("deepocsort_min_hits", 1)),
            delta_t=int(cfg.get("deepocsort_delta_t", 3)),
            inertia=float(cfg.get("deepocsort_inertia", 0.2)),
        )

    cap = cv2.VideoCapture(str(cfg["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {cfg['video']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps else 0.0

    warmup_info = {"enabled": False, "warmup_frames": 0, "warmup_runtime_s": 0.0}
    if device == "mps" and int(cfg.get("warmup_frames", 0)) > 0:
        ok, warmup_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read first frame for warmup: {cfg['video']}")
        warmup_info = warmup_model(model, warmup_frame, cfg, device)
        cap.release()
        cap = cv2.VideoCapture(str(cfg["video"]))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to reopen video after warmup: {cfg['video']}")

    output_dir = build_output_dir(cfg)
    detections_csv = output_dir / "detections.csv"
    events_csv = output_dir / "track_events.csv"
    merged_events_csv = output_dir / "track_events_merged.csv"
    summary_json = output_dir / "summary.json"
    annotated_video = output_dir / f"{cfg['video'].stem}_annotated.mp4"

    writer = None
    if cfg["save_video"]:
        out_fps = output_video_fps(fps, int(cfg["stride"]), str(cfg.get("render_mode", "analysis")))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(annotated_video), fourcc, out_fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_video), fourcc, out_fps, (width, height))

    detections = []
    events = {}
    detection_counts = Counter()
    unique_counts = Counter()
    processed_frames = 0
    frame_idx = -1
    start_time = time.time()
    frame_inference_times = []
    render_cache = {}
    last_overlay_items = []
    stride_val = max(int(cfg["stride"]), 1)
    synthetic_track_id = 1_000_000
    untracked_assignment_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        timestamp_s = frame_idx / fps if fps else 0.0
        is_inference_frame = (frame_idx % stride_val == 0)

        if not is_inference_frame:
            if writer is not None and render_mode_writes_non_inference_frames(str(cfg.get("render_mode", "analysis"))):
                for item in last_overlay_items:
                    draw_box(frame, item["xyxy"], item["color"], item["label"])
                overlay = [
                    f"device={device}",
                    f"tracker={tracker_backend}",
                    f"frame={frame_idx}/{total_frames}",
                    f"time={timestamp_s:.2f}s",
                    f"unique={sum(unique_counts.values())}",
                    f"dets={sum(detection_counts.values())}",
                ]
                for overlay_idx, text in enumerate(overlay):
                    cv2.putText(
                        frame,
                        text,
                        (16, 28 + overlay_idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(frame)
            continue

        processed_frames += 1
        infer_start = time.time()
        if tracker_backend in {"bytetrack", "botsort"}:
            result = model.track(
                frame,
                persist=True,
                tracker=str(tracker_cfg_path),
                conf=float(cfg["conf"]),
                iou=float(cfg["iou"]),
                imgsz=int(cfg["imgsz"]),
                device=device,
                verbose=False,
            )[0]
            frame_inference_times.append(time.time() - infer_start)
            frame_detections = extract_predict_detections(result)
            track_ids = result.boxes.id.int().cpu().tolist() if result.boxes is not None and result.boxes.id is not None else []
            if frame_detections and len(track_ids) != len(frame_detections):
                track_ids = list(range(synthetic_track_id, synthetic_track_id + len(frame_detections)))
                synthetic_track_id += len(frame_detections)
                untracked_assignment_count += len(frame_detections)
        else:
            result = model.predict(
                frame,
                conf=float(cfg["conf"]),
                iou=float(cfg["iou"]),
                imgsz=int(cfg["imgsz"]),
                device=device,
                verbose=False,
            )[0]
            frame_inference_times.append(time.time() - infer_start)
            frame_detections = extract_predict_detections(result)
            track_ids = deepocsort.update(frame_detections, frame, frame_idx)

        current_render_keys = set()
        if frame_detections and len(track_ids) != len(frame_detections):
            raise RuntimeError(
                f"Tracker backend {tracker_backend} returned mismatched assignments: "
                f"{len(track_ids)} ids for {len(frame_detections)} detections"
            )

        for det, track_id in zip(frame_detections, track_ids):
            cls_id = int(det["class_id"])
            cls_name = names[cls_id]
            x1, y1, x2, y2 = [float(v) for v in det["xyxy"]]
            width_px = max(x2 - x1, 0.0)
            height_px = max(y2 - y1, 0.0)
            bbox_area_px2 = width_px * height_px
            event_key = f"{cls_id}:{track_id}"
            event_id = f"{cls_name}-{track_id}"
            current_render_keys.add(event_key)

            detections.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": round(timestamp_s, 3),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(det["confidence"]), 4),
                    "track_id": int(track_id),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width_px": round(width_px, 2),
                    "height_px": round(height_px, 2),
                    "bbox_area_px2": round(bbox_area_px2, 2),
                }
            )
            detection_counts[cls_name] += 1

            if event_key not in events:
                events[event_key] = {
                    "event_id": event_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "track_id": int(track_id),
                    "first_frame": frame_idx,
                    "first_timestamp_s": timestamp_s,
                    "last_frame": frame_idx,
                    "last_timestamp_s": timestamp_s,
                    "num_hits": 1,
                    "max_confidence": float(det["confidence"]),
                    "best_bbox_xyxy": [x1, y1, x2, y2],
                }
                unique_counts[cls_name] += 1
            else:
                event = events[event_key]
                event["last_frame"] = frame_idx
                event["last_timestamp_s"] = timestamp_s
                event["num_hits"] += 1
                if float(det["confidence"]) > event["max_confidence"]:
                    event["max_confidence"] = float(det["confidence"])
                    event["best_bbox_xyxy"] = [x1, y1, x2, y2]

            label = f"{cls_name} conf={det['confidence']:.2f} id={track_id}"
            render_cache[event_key] = {
                "xyxy": det["xyxy"],
                "color": CLASS_COLORS.get(cls_id, (255, 255, 255)),
                "label": label,
                "last_seen_frame": frame_idx,
            }
            draw_box(frame, det["xyxy"], CLASS_COLORS.get(cls_id, (255, 255, 255)), label)

        stale_items = stale_render_items(
            render_cache,
            current_render_keys,
            frame_idx,
            int(cfg.get("render_hold_frames", 0)),
        )
        for item in stale_items:
            draw_box(frame, item["xyxy"], item["color"], item["label"])

        last_overlay_items = [{"xyxy": item["xyxy"], "color": item["color"], "label": item["label"]} for item in render_cache.values()]

        overlay = [
            f"device={device}",
            f"tracker={tracker_backend}",
            f"frame={frame_idx}/{total_frames}",
            f"time={timestamp_s:.2f}s",
            f"unique={sum(unique_counts.values())}",
            f"dets={sum(detection_counts.values())}",
        ]
        for overlay_idx, text in enumerate(overlay):
            cv2.putText(frame, text, (16, 28 + overlay_idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)
        if int(cfg["max_frames"]) > 0 and processed_frames >= int(cfg["max_frames"]):
            break

    cap.release()
    if writer is not None:
        writer.release()

    runtime_s = time.time() - start_time
    infer_fps = processed_frames / runtime_s if runtime_s > 0 else 0.0
    avg_frame_inference_s = sum(frame_inference_times) / len(frame_inference_times) if frame_inference_times else 0.0
    avg_post_first_frame_inference_s = (
        sum(frame_inference_times[1:]) / len(frame_inference_times[1:]) if len(frame_inference_times) > 1 else 0.0
    )

    event_rows = build_event_rows(events)
    write_csv(detections_csv, detections)
    write_csv(events_csv, event_rows)
    raw_metrics = compute_event_metrics(
        event_rows,
        gap_frames=int(cfg.get("fragmentation_gap_frames", 15)),
        iou_threshold=float(cfg.get("fragmentation_iou_threshold", 0.5)),
    )

    homography_enabled = False
    merged_rows = []
    final_rows = event_rows
    if cfg["dedup_mode"] == "track_plus_plane":
        homography_cfg = load_homography_config(cfg["homography_config"])
        merged_rows = merge_events_by_plane(event_rows, homography_cfg)
        write_csv(merged_events_csv, merged_rows)
        final_rows = merged_rows
        homography_enabled = True

    final_metrics = compute_event_metrics(
        final_rows,
        gap_frames=int(cfg.get("fragmentation_gap_frames", 15)),
        iou_threshold=float(cfg.get("fragmentation_iou_threshold", 0.5)),
    )

    summary = {
        "video": str(cfg["video"]),
        "weights": str(cfg["weights"]),
        "repo_root": str(cfg["repo_root"]),
        "tracker_backend": tracker_backend,
        "tracker_requested": tracker_yaml_name or tracker_backend,
        "tracker_actual": tracker_backend if tracker_backend == "deepocsort" else tracker_yaml_name,
        "dedup_mode": cfg["dedup_mode"],
        "homography_enabled": homography_enabled,
        "device_requested": cfg["device"],
        "device_actual": device,
        "mps_status": mps_status,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 3),
        "processed_frames": processed_frames,
        "sampling_stride": int(cfg["stride"]),
        "render_mode": str(cfg.get("render_mode", "analysis")),
        "runtime_s": round(runtime_s, 3),
        "inference_fps": round(infer_fps, 3),
        "first_processed_frame_inference_s": round(frame_inference_times[0], 3) if frame_inference_times else 0.0,
        "avg_frame_inference_s": round(avg_frame_inference_s, 3),
        "avg_post_first_frame_inference_s": round(avg_post_first_frame_inference_s, 3),
        "warmup": warmup_info,
        "render_hold_frames": int(cfg.get("render_hold_frames", 0)),
        "total_detections": len(detections),
        "untracked_assignment_count": untracked_assignment_count,
        "unique_track_events_raw": len(event_rows),
        "unique_track_events": len(final_rows),
        "mean_hits_per_event_raw": raw_metrics["mean_hits_per_event"],
        "median_event_duration_frames_raw": raw_metrics["median_event_duration_frames"],
        "mean_event_duration_frames_raw": raw_metrics["mean_event_duration_frames"],
        "fragmentation_proxy_raw": raw_metrics["fragmentation_proxy"],
        "mean_hits_per_event": final_metrics["mean_hits_per_event"],
        "median_event_duration_frames": final_metrics["median_event_duration_frames"],
        "mean_event_duration_frames": final_metrics["mean_event_duration_frames"],
        "fragmentation_proxy": final_metrics["fragmentation_proxy"],
        "detection_counts_by_class": dict(detection_counts),
        "unique_counts_by_class": dict(unique_counts),
        "notes": summary_note_block(),
        "outputs": {
            "detections_csv": str(detections_csv),
            "track_events_csv": str(events_csv),
            "track_events_merged_csv": str(merged_events_csv) if merged_rows else "",
            "summary_json": str(summary_json),
            "annotated_video": str(annotated_video) if cfg["save_video"] else "",
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def compare_trackers(cfg: dict):
    comparison_rows = []
    for backend in ("bytetrack", "botsort", "deepocsort"):
        backend_cfg = dict(cfg)
        backend_cfg["tracker_backend"] = backend
        backend_cfg["dedup_mode"] = "track_only"
        tracker_yaml = tracker_backend_to_yaml(backend)
        if tracker_yaml:
            backend_cfg["tracker"] = tracker_yaml
        summary = run_single(backend_cfg)
        comparison_rows.append(
            {
                "tracker_backend": summary["tracker_backend"],
                "runtime_s": summary["runtime_s"],
                "inference_fps": summary["inference_fps"],
                "first_processed_frame_inference_s": summary["first_processed_frame_inference_s"],
                "avg_post_first_frame_inference_s": summary["avg_post_first_frame_inference_s"],
                "total_detections": summary["total_detections"],
                "unique_track_events": summary["unique_track_events"],
                "mean_hits_per_event": summary["mean_hits_per_event"],
                "median_event_duration_frames": summary["median_event_duration_frames"],
                "mean_event_duration_frames": summary["mean_event_duration_frames"],
                "fragmentation_proxy": summary["fragmentation_proxy"],
                "summary_json": summary["outputs"]["summary_json"],
            }
        )

    selected_backend, selection_reason = select_default_tracker(comparison_rows)
    for row in comparison_rows:
        row["selected_default_tracker"] = selected_backend
        row["selection_reason"] = selection_reason
        row["is_selected"] = row["tracker_backend"] == selected_backend

    comparison_dir = ensure_dir(cfg["output_root"] / "comparisons")
    comparison_csv = comparison_dir / "public_benchmark_summary.csv"
    write_csv(comparison_csv, comparison_rows)
    print(
        json.dumps(
            {
                "comparison_csv": str(comparison_csv),
                "selected_default_tracker": selected_backend,
                "selection_reason": selection_reason,
                "rows": comparison_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main():
    cfg = merge_config(parse_args())
    if cfg["compare_trackers"]:
        compare_trackers(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
