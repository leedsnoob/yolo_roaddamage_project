#!/usr/bin/env python3
"""Run a lightweight end-to-end smoke test for the packaged thesis pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
OUTPUT_ROOT = PIPELINE_ROOT / "outputs" / "smoke_test"


def run_step(name: str, command: list[str]) -> dict:
    started = command[:]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    return {
        "name": name,
        "command": started,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def assert_step(step: dict) -> None:
    if step["returncode"] != 0:
        print(json.dumps(step, indent=2, ensure_ascii=False))
        raise RuntimeError(f"Smoke step failed: {step['name']}")


def validate_samples() -> dict:
    sample_root = PIPELINE_ROOT / "02_detection" / "samples" / "images"
    selected_csv = sample_root / "selected_images.csv"
    label_dir = sample_root / "labels"
    rows = list(csv.DictReader(selected_csv.open(newline="", encoding="utf-8")))
    drone = [row["image_name"] for row in rows if "Drone" in row["image_name"]]
    empty = []
    missing = []
    for row in rows:
        label_path = label_dir / f"{Path(row['image_name']).stem}.txt"
        if not label_path.exists():
            missing.append(row["image_name"])
            continue
        if not label_path.read_text(encoding="utf-8").strip():
            empty.append(row["image_name"])
    return {
        "selected_csv": str(selected_csv),
        "num_images": len(rows),
        "drone_images": drone,
        "missing_labels": missing,
        "empty_labels": empty,
        "is_non_drone_non_empty": not drone and not missing and not empty,
    }


def check_area_assets() -> dict:
    area_root = PIPELINE_ROOT / "04_area_measurement" / "assets" / "area"
    wide_csv = area_root / "four_method_area_results_wide.csv"
    long_csv = area_root / "four_method_area_results_long.csv"
    wide_rows = list(csv.DictReader(wide_csv.open(newline="", encoding="utf-8")))
    long_rows = list(csv.DictReader(long_csv.open(newline="", encoding="utf-8")))
    required = {"M1_empirical_bbox_m2", "M3_depth_anything_v2_m2", "M4_metric3d_m2"}
    return {
        "wide_csv": str(wide_csv),
        "long_csv": str(long_csv),
        "wide_rows": len(wide_rows),
        "long_rows": len(long_rows),
        "has_report_methods": bool(wide_rows) and required.issubset(wide_rows[0].keys()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick packaged road-damage pipeline smoke test.")
    parser.add_argument("--device", default="cpu", help="Use cpu by default for deterministic smoke tests.")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--max-video-frames", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    sample_validation = validate_samples()
    if not sample_validation["is_non_drone_non_empty"]:
        raise RuntimeError(f"Packaged samples are not clean non-drone non-empty: {sample_validation}")

    area_assets = check_area_assets()
    if not area_assets["has_report_methods"]:
        raise RuntimeError(f"Area assets missing required M1/M3/M4 columns: {area_assets}")

    step = run_step(
        "02_detection_sample_inference",
        [
            sys.executable,
            "road_damage_pipeline/02_detection/scripts/infer_images.py",
            "--device",
            args.device,
            "--max-images",
            str(args.max_images),
            "--output-root",
            str(OUTPUT_ROOT / "02_detection"),
        ],
    )
    assert_step(step)
    steps.append(step)

    video_output_root = OUTPUT_ROOT / "03_video_dedup"
    step = run_step(
        "03_video_dedup_short_inference",
        [
            sys.executable,
            "road_damage_pipeline/03_video_dedup/scripts/infer_video.py",
            "--device",
            args.device,
            "--max-frames",
            str(args.max_video_frames),
            "--output-root",
            str(video_output_root),
        ],
    )
    assert_step(step)
    steps.append(step)

    video_results_dir = video_output_root / "3_dense_130_190" / "bytetrack" / "track_only"
    if not (video_results_dir / "summary.json").exists():
        raise RuntimeError(f"Video dedup output missing summary.json: {video_results_dir}")

    step = run_step(
        "05_report_generation_image",
        [
            sys.executable,
            "road_damage_pipeline/05_report_generation/scripts/report_pipeline.py",
            "--mode",
            "image",
            "--device",
            args.device,
            "--max-images",
            "1",
            "--output-root",
            str(OUTPUT_ROOT / "05_report_generation"),
        ],
    )
    assert_step(step)
    steps.append(step)

    step = run_step(
        "05_report_generation_video",
        [
            sys.executable,
            "road_damage_pipeline/05_report_generation/scripts/report_pipeline.py",
            "--mode",
            "video",
            "--device",
            args.device,
            "--video-results-dir",
            str(video_results_dir),
            "--output-root",
            str(OUTPUT_ROOT / "05_report_generation"),
        ],
    )
    assert_step(step)
    steps.append(step)

    summary = {
        "repo_root": str(REPO_ROOT),
        "sample_validation": sample_validation,
        "area_assets": area_assets,
        "video_results_dir": str(video_results_dir),
        "steps": steps,
    }
    summary_path = OUTPUT_ROOT / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(summary_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
