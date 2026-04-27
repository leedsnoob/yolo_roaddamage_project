import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
import yaml


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_bbox_value(value):
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]
    raise TypeError(f"Unsupported bbox value: {type(value)!r}")


def event_duration_frames(row):
    return int(row["last_frame"]) - int(row["first_frame"]) + 1


def compute_fragmentation_proxy(event_rows, gap_frames=15, iou_threshold=0.5):
    if not event_rows:
        return 0

    sorted_rows = sorted(
        event_rows,
        key=lambda row: (
            int(row["first_frame"]),
            int(row["class_id"]),
            str(row.get("track_id", row.get("merged_event_id", "0"))),
        ),
    )
    proxy = 0
    for idx, left in enumerate(sorted_rows):
        left_box = parse_bbox_value(left["best_bbox_xyxy"])
        for right in sorted_rows[idx + 1 :]:
            if int(left["class_id"]) != int(right["class_id"]):
                continue
            frame_gap = int(right["first_frame"]) - int(left["last_frame"])
            if frame_gap < 0:
                frame_gap = int(left["first_frame"]) - int(right["last_frame"])
            if frame_gap > gap_frames:
                break
            right_box = parse_bbox_value(right["best_bbox_xyxy"])
            if box_iou_xyxy(left_box, right_box) >= iou_threshold:
                proxy += 1
    return proxy


def compute_event_metrics(event_rows, gap_frames=15, iou_threshold=0.5):
    if not event_rows:
        return {
            "mean_hits_per_event": 0.0,
            "median_event_duration_frames": 0.0,
            "mean_event_duration_frames": 0.0,
            "fragmentation_proxy": 0,
        }

    hits = [int(row["num_hits"]) for row in event_rows]
    durations = [event_duration_frames(row) for row in event_rows]
    return {
        "mean_hits_per_event": round(mean(hits), 3),
        "median_event_duration_frames": round(float(median(durations)), 3),
        "mean_event_duration_frames": round(mean(durations), 3),
        "fragmentation_proxy": int(compute_fragmentation_proxy(event_rows, gap_frames=gap_frames, iou_threshold=iou_threshold)),
    }


def load_homography_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    matrix = cfg.get("matrix")
    if matrix is not None:
        H = matrix
    else:
        src_points = cfg.get("src_points")
        dst_points = cfg.get("dst_points")
        if not src_points or not dst_points or len(src_points) != len(dst_points):
            raise ValueError(f"Homography config must define either `matrix` or matching `src_points`/`dst_points`: {path}")
        H, _ = cv2.findHomography(
            srcPoints=np.array(src_points, dtype=float),
            dstPoints=np.array(dst_points, dtype=float),
        )
        H = H.tolist()

    return {
        "matrix": H,
        "merge_distance_thresh": float(cfg.get("merge_distance_thresh", 60.0)),
        "merge_time_gap_frames": int(cfg.get("merge_time_gap_frames", 30)),
        "max_size_ratio": float(cfg.get("max_size_ratio", 4.0)),
        "max_aspect_ratio_delta": float(cfg.get("max_aspect_ratio_delta", 2.0)),
    }


def project_point_homography(point_xy, matrix):
    x, y = point_xy
    px = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    py = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    pw = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(pw) < 1e-6:
        return [x, y]
    return [px / pw, py / pw]


def event_anchor_and_shape(row, matrix):
    box = parse_bbox_value(row["best_bbox_xyxy"])
    x1, y1, x2, y2 = box
    anchor = [(x1 + x2) / 2.0, y2]
    plane_anchor = project_point_homography(anchor, matrix)
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return {
        "plane_anchor": plane_anchor,
        "area": width * height,
        "aspect_ratio": width / height,
    }


def should_merge_events(left, right, homography_cfg):
    if int(left["class_id"]) != int(right["class_id"]):
        return False

    first_left, last_left = int(left["first_frame"]), int(left["last_frame"])
    first_right, last_right = int(right["first_frame"]), int(right["last_frame"])
    frame_gap = max(first_right - last_left, first_left - last_right, 0)
    if frame_gap > homography_cfg["merge_time_gap_frames"]:
        return False

    left_meta = event_anchor_and_shape(left, homography_cfg["matrix"])
    right_meta = event_anchor_and_shape(right, homography_cfg["matrix"])
    dx = left_meta["plane_anchor"][0] - right_meta["plane_anchor"][0]
    dy = left_meta["plane_anchor"][1] - right_meta["plane_anchor"][1]
    if math.hypot(dx, dy) > homography_cfg["merge_distance_thresh"]:
        return False

    size_ratio = max(left_meta["area"], right_meta["area"]) / max(min(left_meta["area"], right_meta["area"]), 1e-6)
    if size_ratio > homography_cfg["max_size_ratio"]:
        return False

    aspect_ratio_ratio = max(left_meta["aspect_ratio"], right_meta["aspect_ratio"]) / max(
        min(left_meta["aspect_ratio"], right_meta["aspect_ratio"]), 1e-6
    )
    if aspect_ratio_ratio > homography_cfg["max_aspect_ratio_delta"]:
        return False

    return True


def merge_events_by_plane(event_rows, homography_cfg):
    if not event_rows:
        return []

    rows = sorted(event_rows, key=lambda row: (int(row["first_frame"]), int(row["class_id"]), int(row["track_id"])))
    parent = list(range(len(rows)))

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left_idx, right_idx):
        left_root, right_root = find(left_idx), find(right_idx)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_idx, left in enumerate(rows):
        for right_idx in range(left_idx + 1, len(rows)):
            right = rows[right_idx]
            if int(left["class_id"]) != int(right["class_id"]):
                continue
            if should_merge_events(left, right, homography_cfg):
                union(left_idx, right_idx)

    groups = {}
    for idx, row in enumerate(rows):
        groups.setdefault(find(idx), []).append(row)

    merged_rows = []
    for group_idx, members in enumerate(groups.values(), start=1):
        members = sorted(members, key=lambda row: (int(row["first_frame"]), int(row["track_id"])))
        best = max(members, key=lambda row: float(row["max_confidence"]))
        class_name = members[0]["class_name"]
        class_id = int(members[0]["class_id"])
        track_ids = [int(row["track_id"]) for row in members]
        source_event_ids = [row["event_id"] for row in members]
        first_row = min(members, key=lambda row: int(row["first_frame"]))
        last_row = max(members, key=lambda row: int(row["last_frame"]))
        plane_points = [event_anchor_and_shape(member, homography_cfg["matrix"])["plane_anchor"] for member in members]
        plane_x = round(mean(point[0] for point in plane_points), 3)
        plane_y = round(mean(point[1] for point in plane_points), 3)
        merged_rows.append(
            {
                "merged_event_id": f"{class_name}-merged-{group_idx}",
                "class_id": class_id,
                "class_name": class_name,
                "track_ids": json.dumps(track_ids, ensure_ascii=False),
                "source_event_ids": json.dumps(source_event_ids, ensure_ascii=False),
                "first_frame": int(first_row["first_frame"]),
                "first_timestamp_s": float(first_row["first_timestamp_s"]),
                "last_frame": int(last_row["last_frame"]),
                "last_timestamp_s": float(last_row["last_timestamp_s"]),
                "duration_s": round(float(last_row["last_timestamp_s"]) - float(first_row["first_timestamp_s"]), 3),
                "num_hits": sum(int(row["num_hits"]) for row in members),
                "max_confidence": max(float(row["max_confidence"]) for row in members),
                "best_bbox_xyxy": best["best_bbox_xyxy"],
                "plane_anchor_x": plane_x,
                "plane_anchor_y": plane_y,
                "num_source_events": len(members),
            }
        )
    merged_rows.sort(key=lambda row: (row["first_frame"], row["class_id"], row["merged_event_id"]))
    return merged_rows


def select_default_tracker(summary_rows):
    if not summary_rows:
        return None, "no summary rows"

    baseline = next((row for row in summary_rows if row["tracker_backend"] == "bytetrack"), None)
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            float(row["fragmentation_proxy"]),
            -float(row["mean_hits_per_event"]),
            -float(row["inference_fps"]),
        ),
    )
    if baseline is None:
        return ranked[0]["tracker_backend"], "bytetrack baseline missing; selected best available by metric ordering"

    baseline_fragmentation = float(baseline["fragmentation_proxy"])
    if baseline_fragmentation <= 0:
        tied_rows = [row for row in ranked if float(row["fragmentation_proxy"]) <= 0]
        best_tied = sorted(
            tied_rows,
            key=lambda row: (-float(row["mean_hits_per_event"]), -float(row["inference_fps"])),
        )[0]
        return best_tied["tracker_backend"], "bytetrack baseline already had zero fragmentation; selected by mean_hits/FPS tie-break"

    for row in ranked:
        if row["tracker_backend"] == "bytetrack":
            continue
        frag_ok = float(row["fragmentation_proxy"]) <= baseline_fragmentation * 0.9
        fps_ok = float(row["inference_fps"]) >= float(baseline["inference_fps"]) * 0.75
        if frag_ok and fps_ok:
            return row["tracker_backend"], "selected by >=10% lower fragmentation with <=25% FPS loss versus bytetrack"

    return "bytetrack", "no alternative met the fragmentation/FPS threshold versus bytetrack"


def write_csv(path, rows):
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
