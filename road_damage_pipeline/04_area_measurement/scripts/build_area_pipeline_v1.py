#!/usr/bin/env python3
"""Build the final area-estimation pipeline evidence package.

This script does not re-run FastSAM, Depth Anything V2, or Metric3D. It
combines the already generated experiment outputs into one pipeline artifact
for the thesis/demo frontend:

M1: empirical bbox rule
M2: FastSAM mask from bbox crop with class-derived damage text prompt
M3: Depth Anything V2 + senior empirical bbox prior
M4: Metric3D + senior empirical bbox prior
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MODULE_ROOT.parents[1]
AREA_ROOT = WORKSPACE_ROOT / "area_experiments"
V1_ROOT = AREA_ROOT / "area_measurement_v1"
M123_CSV = V1_ROOT / "area_results.csv"
M4_CSV = V1_ROOT / "metric3d_trial" / "metric3d_area_results.csv"
INPUT_CSV = AREA_ROOT / "selected_rdd_test_lane_countries" / "final_selection" / "gt_bboxes_from_txt.csv"
RAW_DIR = AREA_ROOT / "selected_rdd_test_lane_countries" / "final_selection" / "raw"
M123_VIS_DIR = V1_ROOT / "visuals"
M4_VIS_DIR = V1_ROOT / "metric3d_trial" / "visuals"
OUT_ROOT = MODULE_ROOT / "assets" / "area"

METHOD_ORDER = [
    "M1_empirical_bbox",
    "M2_fastsam_mask",
    "M3_depth_anything_v2_empirical_bbox",
    "M4_metric3d_empirical_bbox",
]
METHOD_SHORT = {
    "M1_empirical_bbox": "M1 Empirical bbox",
    "M2_fastsam_mask": "M2 FastSAM mask",
    "M3_depth_anything_v2_empirical_bbox": "M3 Depth Anything V2",
    "M4_metric3d_empirical_bbox": "M4 Metric3D",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_packaged_assets() -> dict[str, object]:
    required_files = [
        OUT_ROOT / "four_method_area_results_long.csv",
        OUT_ROOT / "four_method_area_results_wide.csv",
        OUT_ROOT / "four_method_area_summary_by_class.csv",
        OUT_ROOT / "manifest.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw area experiment files are not available and packaged area assets are incomplete: "
            + "; ".join(missing)
        )

    long_rows = read_csv(OUT_ROOT / "four_method_area_results_long.csv")
    wide_rows = read_csv(OUT_ROOT / "four_method_area_results_wide.csv")
    methods = sorted({row["method"] for row in long_rows})
    required_wide_columns = {
        "M1_empirical_bbox_m2",
        "M2_fastsam_mask_m2",
        "M3_depth_anything_v2_m2",
        "M4_metric3d_m2",
    }
    has_required_columns = bool(wide_rows) and required_wide_columns.issubset(wide_rows[0].keys())
    if not has_required_columns:
        raise ValueError("Packaged wide area CSV is missing one or more M1/M2/M3/M4 columns.")

    manifest = json.loads((OUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return {
        "status": "packaged_assets_validated",
        "reason": "raw area_experiments directory was not found; using repository-packaged evidence",
        "output_root": str(OUT_ROOT),
        "num_gt_boxes": len(wide_rows),
        "num_method_rows": len(long_rows),
        "methods": methods,
        "manifest": manifest,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    return float(text)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def row_key(row: dict[str, str]) -> tuple[str, int]:
    return row["image_name"], int(row["box_index"])


def format_area(value: object) -> str:
    text = str(value).strip()
    if text == "":
        return "-"
    return f"{float(text):.3f}"


def combine_results() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    m123 = read_csv(M123_CSV)
    m4 = read_csv(M4_CSV)
    all_rows = [row for row in m123 + m4 if row["method"] in METHOD_ORDER]
    all_rows.sort(key=lambda row: (row["image_name"], int(row["box_index"]), METHOD_ORDER.index(row["method"])))

    long_rows: list[dict[str, object]] = []
    for row in all_rows:
        long_rows.append(
            {
                "image_name": row["image_name"],
                "country": row["country"],
                "box_index": int(row["box_index"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "method": row["method"],
                "method_label": METHOD_SHORT[row["method"]],
                "status": row["status"],
                "estimated_area_m2": row["estimated_area_m2"],
                "width_m": row["width_m"],
                "height_m": row["height_m"],
                "mask_pixels": row.get("mask_pixels", ""),
                "fallback_used": row.get("fallback_used", ""),
                "fallback_reason": row.get("fallback_reason", ""),
                "depth_median_m": row.get("depth_median_m", ""),
                "depth_area_is_assumption": row.get("depth_area_is_assumption", ""),
                "fastsam_prompt": row.get("fastsam_prompt", ""),
                "mask_ratio": row.get("mask_ratio", ""),
                "raw_depth_bbox_area_m2": row.get("raw_depth_bbox_area_m2", ""),
                "empirical_prior_area_m2": row.get("empirical_prior_area_m2", ""),
                "method_notes": row.get("method_notes", ""),
            }
        )

    grouped: dict[tuple[str, int], dict[str, str]] = {}
    meta: dict[tuple[str, int], dict[str, str]] = {}
    for row in all_rows:
        key = row_key(row)
        grouped.setdefault(key, {})[row["method"]] = row["estimated_area_m2"]
        meta.setdefault(key, row)

    wide_rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        row = meta[key]
        values = grouped[key]
        m1 = as_float(values.get("M1_empirical_bbox"), 0.0)
        m2 = as_float(values.get("M2_fastsam_mask"), 0.0)
        m3 = as_float(values.get("M3_depth_anything_v2_empirical_bbox"), 0.0)
        m4 = as_float(values.get("M4_metric3d_empirical_bbox"), 0.0)
        wide_rows.append(
            {
                "image_name": row["image_name"],
                "country": row["country"],
                "box_index": int(row["box_index"]),
                "class_name": row["class_name"],
                "bbox_width_px": row["bbox_width_px"],
                "bbox_height_px": row["bbox_height_px"],
                "M1_empirical_bbox_m2": f"{m1:.6f}",
                "M2_fastsam_mask_m2": f"{m2:.6f}",
                "M3_depth_anything_v2_m2": f"{m3:.6f}",
                "M4_metric3d_m2": f"{m4:.6f}",
                "M2_over_M1": "" if m1 <= 0 else f"{m2 / m1:.3f}",
                "M3_over_M1": "" if m1 <= 0 else f"{m3 / m1:.3f}",
                "M4_over_M1": "" if m1 <= 0 else f"{m4 / m1:.3f}",
            }
        )
    return long_rows, wide_rows


def summarize_by_class(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        grouped[(str(row["class_name"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for class_name, method in sorted(grouped, key=lambda item: (item[0], METHOD_ORDER.index(item[1]))):
        rows = grouped[(class_name, method)]
        areas = [as_float(row["estimated_area_m2"]) for row in rows if str(row["estimated_area_m2"]).strip()]
        summary_rows.append(
            {
                "class_name": class_name,
                "method": method,
                "method_label": METHOD_SHORT[method],
                "num_boxes": len(rows),
                "success_count": len(areas),
                "fallback_count": sum(as_bool(row["fallback_used"]) for row in rows),
                "assumption_count": sum(as_bool(row["depth_area_is_assumption"]) for row in rows),
                "mean_area_m2": "" if not areas else f"{mean(areas):.6f}",
                "median_area_m2": "" if not areas else f"{median(areas):.6f}",
                "min_area_m2": "" if not areas else f"{min(areas):.6f}",
                "max_area_m2": "" if not areas else f"{max(areas):.6f}",
            }
        )
    return summary_rows


def wrap_text(line: str, max_chars: int) -> list[str]:
    if not line:
        return [""]
    words = line.split()
    wrapped: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                wrapped.append(current)
            current = word
    if current:
        wrapped.append(current)
    return wrapped


def draw_text_panel(lines: list[str], size: tuple[int, int] = (780, 520)) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    y = 42
    for idx, line in enumerate(lines):
        font_scale = 0.72 if idx == 0 else 0.54
        thickness = 2 if idx == 0 else 1
        max_chars = 58 if idx == 0 else 72
        for segment in wrap_text(line, max_chars):
            cv2.putText(panel, segment, (24, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), thickness, cv2.LINE_AA)
            y += 38 if idx == 0 else 28
            if y > height - 24:
                return panel
    return panel


def resize_keep(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_combined_boards(wide_rows: list[dict[str, object]]) -> list[str]:
    out_dir = ensure_dir(OUT_ROOT / "visuals" / "four_method_boards")
    paths: list[str] = []
    rows_by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in wide_rows:
        rows_by_image[str(row["image_name"])].append(row)

    for image_name, rows in sorted(rows_by_image.items()):
        stem = Path(image_name).stem
        area_board_path = M123_VIS_DIR / f"{stem}_area_board.jpg"
        metric_board_path = M4_VIS_DIR / f"{stem}_metric3d_board.jpg"
        raw_path = RAW_DIR / image_name

        raw = cv2.imread(str(raw_path))
        area_board = cv2.imread(str(area_board_path))
        metric_board = cv2.imread(str(metric_board_path))
        if raw is None or area_board is None or metric_board is None:
            continue

        raw = resize_keep(raw, (780, 520))
        area_board = resize_keep(area_board, (780, 520))
        metric_board = resize_keep(metric_board, (780, 520))

        lines = [f"{stem}: four-method visual evidence"]
        for row in sorted(rows, key=lambda item: int(item["box_index"])):
            lines.append(
                f"{row['class_name']}#{row['box_index']} | M1 bbox | M2 FastSAM crop | M3/M4 bbox-depth ratio"
            )
        lines.extend(
            [
                "",
                "M1: bbox empirical rule with fixed scale.",
                "M2: crop GT bbox, then use FastSAM with a coarse damage text prompt.",
                "M3/M4: sample depth inside the bbox and apply the senior empirical bbox ratio.",
                "Visual explanation only. Numeric areas are stored in CSV tables.",
            ]
        )
        text_panel = draw_text_panel(lines, (780, 520))
        top = np.concatenate([raw, text_panel], axis=1)
        bottom = np.concatenate([area_board, metric_board], axis=1)
        board = np.concatenate([top, bottom], axis=0)
        out_path = out_dir / f"{stem}_four_method_board.jpg"
        cv2.imwrite(str(out_path), board)
        paths.append(str(out_path))
    return paths


def make_summary_chart(wide_rows: list[dict[str, object]], out_path: Path) -> None:
    width, height = 1800, 980
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Area estimation comparison on selected RDD GT boxes", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Values are estimated m2. M3/M4 use monocular depth with assumed camera/FOV.", (40, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 80), 1, cv2.LINE_AA)

    col_x = [44, 430, 570, 720, 875, 1030, 1190, 1345, 1500]
    headers = ["image", "cls", "box", "M1", "M2", "M3", "M4", "M3/M1", "M4/M1"]
    for x, header in zip(col_x, headers):
        cv2.putText(canvas, header, (x, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 70, 70), 2, cv2.LINE_AA)
    y = 175
    for row in wide_rows:
        values = [
            Path(str(row["image_name"])).stem[:32],
            str(row["class_name"]),
            str(row["box_index"]),
            format_area(row["M1_empirical_bbox_m2"]),
            format_area(row["M2_fastsam_mask_m2"]),
            format_area(row["M3_depth_anything_v2_m2"]),
            format_area(row["M4_metric3d_m2"]),
            str(row["M3_over_M1"]),
            str(row["M4_over_M1"]),
        ]
        for x, value in zip(col_x, values):
            cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)
        y += 48
        if y > height - 60:
            break
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


def write_readme(long_rows: list[dict[str, object]], wide_rows: list[dict[str, object]], summary_rows: list[dict[str, object]], board_paths: list[str]) -> None:
    config = json.loads((V1_ROOT / "config_used.json").read_text(encoding="utf-8"))
    metric_config = json.loads((V1_ROOT / "metric3d_trial" / "config_used.json").read_text(encoding="utf-8"))
    lines = [
        "# Area Measurement Pipeline V1",
        "",
        "This folder combines the existing area-estimation experiments into one pipeline artifact.",
        "The input is the selected RDD test images and their GT bounding boxes from txt labels.",
        "",
        "## Methods",
        "",
        "- M1 empirical bbox: senior class-specific bbox rule with fixed pixel scale.",
        "- M2 FastSAM mask: GT bbox is cropped first; FastSAM receives the crop and a class-derived damage text prompt.",
        "- M3 Depth Anything V2 + empirical bbox: bbox depth area corrected by the senior empirical bbox ratio.",
        "- M4 Metric3D + empirical bbox: bbox depth area corrected by the senior empirical bbox ratio.",
        "",
        "## Parameters",
        "",
        f"- scale_factor_m_per_px: `{config['scale_factor_m_per_px']}`",
        f"- assumed_horizontal_fov_deg: `{config['assumed_horizontal_fov_deg']}`",
        f"- fastsam_model: `{config['fastsam_model']}`",
        f"- Depth Anything V2 model: `{config['depth_model']}`",
        f"- Metric3D model: `{metric_config['metric3d_model']}`",
        f"- Metric3D input size: `{metric_config['metric3d_input_size']}`",
        "",
        "## Outputs",
        "",
        "- `four_method_area_results_long.csv`: one row per bbox per method.",
        "- `four_method_area_results_wide.csv`: one row per bbox with all four method values.",
        "- `four_method_area_summary_by_class.csv`: per-class summary.",
        "- `visuals/four_method_summary_table.png`: compact table for paper/frontend.",
        "- `visuals/four_method_boards/`: per-image boards showing bbox prior, FastSAM crop mask, Depth Anything V2, and Metric3D.",
        "",
        "## Main Caveat",
        "",
        "No camera intrinsics or true physical labels are available. These are estimated areas for method comparison, not ground-truth measurements.",
        "",
        "## Counts",
        "",
        f"- GT boxes: `{len(wide_rows)}`",
        f"- method rows: `{len(long_rows)}`",
        f"- generated per-image boards: `{len(board_paths)}`",
        "",
        "## Quick Result Table",
        "",
        "| class | method | mean area m2 | median area m2 | fallback | assumption |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['class_name']} | {row['method_label']} | {row['mean_area_m2']} | {row['median_area_m2']} | "
            f"{row['fallback_count']} | {row['assumption_count']} |"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not M123_CSV.exists() or not M4_CSV.exists():
        print(json.dumps(validate_packaged_assets(), indent=2, ensure_ascii=False))
        return 0

    ensure_dir(OUT_ROOT)
    long_rows, wide_rows = combine_results()
    summary_rows = summarize_by_class(long_rows)

    write_csv(OUT_ROOT / "four_method_area_results_long.csv", long_rows)
    write_csv(OUT_ROOT / "four_method_area_results_wide.csv", wide_rows)
    write_csv(OUT_ROOT / "four_method_area_summary_by_class.csv", summary_rows)
    board_paths = make_combined_boards(wide_rows)
    make_summary_chart(wide_rows, OUT_ROOT / "visuals" / "four_method_summary_table.png")
    write_readme(long_rows, wide_rows, summary_rows, board_paths)

    manifest = {
        "input_gt_csv": str(INPUT_CSV),
        "source_m1_m2_m3_csv": str(M123_CSV),
        "source_m4_csv": str(M4_CSV),
        "output_root": str(OUT_ROOT),
        "num_gt_boxes": len(wide_rows),
        "num_method_rows": len(long_rows),
        "num_boards": len(board_paths),
        "methods": METHOD_ORDER,
        "visuals": {
            "summary_table": str(OUT_ROOT / "visuals" / "four_method_summary_table.png"),
            "four_method_boards_dir": str(OUT_ROOT / "visuals" / "four_method_boards"),
        },
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
