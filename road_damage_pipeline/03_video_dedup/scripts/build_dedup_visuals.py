#!/usr/bin/env python3
"""Build visual comparison boards for video deduplication results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PIPELINE_ROOT.parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "video_damage_analytics" / "outputs" / "dedup_case_study_yolo11s"
CASE_ROOT = SOURCE_ROOT / "3_dense_130_190"
OUT_DIR = PIPELINE_ROOT / "assets" / "video_dedup"
TRACKERS = ["bytetrack", "botsort", "deepocsort"]
NUM_VISUAL_FRAMES = 4
MIN_FRAME_GAP_S = 8.0


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return frame


def put_title(frame, title: str):
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def clean_old_frame_visuals() -> None:
    for pattern in ("dedup_tracker_comparison_*.jpg",):
        for path in OUT_DIR.glob(pattern):
            path.unlink()
    keyframes_dir = OUT_DIR / "keyframes"
    if keyframes_dir.exists():
        for path in keyframes_dir.glob("*.jpg"):
            path.unlink()


def select_representative_frames(fps: float) -> list[int]:
    counts: Counter[int] = Counter()
    for tracker in TRACKERS:
        detections_path = CASE_ROOT / tracker / "track_only" / "detections.csv"
        df = pd.read_csv(detections_path)
        for frame_idx, count in df.groupby("frame_idx").size().items():
            counts[int(frame_idx)] += int(count)

    min_gap = max(int(round(MIN_FRAME_GAP_S * fps)), 1)
    selected: list[int] = []
    for frame_idx, _ in counts.most_common():
        if all(abs(frame_idx - other) >= min_gap for other in selected):
            selected.append(frame_idx)
        if len(selected) >= NUM_VISUAL_FRAMES:
            break

    if len(selected) < NUM_VISUAL_FRAMES:
        fallback = [int(round(t * fps)) for t in (8, 22, 38, 55)]
        for frame_idx in fallback:
            if all(abs(frame_idx - other) >= min_gap for other in selected):
                selected.append(frame_idx)
            if len(selected) >= NUM_VISUAL_FRAMES:
                break
    return sorted(selected)


def make_frame_comparison_boards():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_old_frame_visuals()
    metadata = json.loads((SOURCE_ROOT / "segments" / "3_dense_130_190.json").read_text(encoding="utf-8"))
    fps = float(metadata["fps"])
    selected_frames = select_representative_frames(fps)
    keyframes_dir = OUT_DIR / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for board_index, frame_idx in enumerate(selected_frames, start=1):
        t = frame_idx / fps
        panels = []
        for tracker in TRACKERS:
            video_path = CASE_ROOT / tracker / "track_only" / "3_dense_130_190_annotated.mp4"
            frame = read_frame(video_path, frame_idx)
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            titled = put_title(frame, f"{tracker} | t={t:.1f}s | frame={frame_idx}")
            panels.append(titled)
            cv2.imwrite(str(keyframes_dir / f"{tracker}_f{frame_idx}_t{t:.1f}s.jpg"), titled)
        board = cv2.hconcat(panels)
        out_path = OUT_DIR / f"dedup_tracker_comparison_{board_index:02d}_f{frame_idx}_t{t:.1f}s.jpg"
        cv2.imwrite(str(out_path), board)
        rows.append(
            {
                "board_index": board_index,
                "time_s": round(t, 3),
                "frame_idx": frame_idx,
                "output": str(out_path.relative_to(PIPELINE_ROOT)),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "dedup_visual_boards.csv", index=False)


def make_event_duration_plot():
    rows = []
    for tracker in TRACKERS:
        events_path = CASE_ROOT / tracker / "track_only" / "track_events.csv"
        df = pd.read_csv(events_path)
        if df.empty:
            continue
        df["duration_frames"] = df["last_frame"] - df["first_frame"] + 1
        df["tracker"] = tracker
        rows.append(df[["tracker", "class_name", "duration_frames", "num_hits", "max_confidence"]])
    all_events = pd.concat(rows, ignore_index=True)
    all_events.to_csv(OUT_DIR / "event_duration_samples.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    all_events.boxplot(column="duration_frames", by="tracker", ax=axes[0], grid=False)
    axes[0].set_title("Track duration distribution")
    axes[0].set_xlabel("Tracker")
    axes[0].set_ylabel("Duration (frames)")
    axes[0].figure.suptitle("")

    all_events.boxplot(column="num_hits", by="tracker", ax=axes[1], grid=False)
    axes[1].set_title("Hits per event distribution")
    axes[1].set_xlabel("Tracker")
    axes[1].set_ylabel("Number of detections")
    axes[1].figure.suptitle("")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "dedup_event_duration_distribution.png")
    plt.close(fig)


def main():
    make_frame_comparison_boards()
    make_event_duration_plot()
    print(f"wrote visuals to {OUT_DIR}")


if __name__ == "__main__":
    main()
