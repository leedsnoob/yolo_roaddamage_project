#!/usr/bin/env python3
"""Summarize packaged final-eval metrics into CSV files for paper writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = PIPELINE_ROOT / "assets" / "final_eval_metrics"
OUT_DIR = PIPELINE_ROOT / "outputs" / "summaries"


def metric_value(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return ""


def main() -> None:
    rows = []
    for path in sorted(METRICS_ROOT.glob("*/metrics_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        overall = data.get("overall", {})
        rows.append(
            {
                "task_name": data.get("task_name", path.parent.name),
                "model_name": data.get("model_name", ""),
                "scale": data.get("scale", ""),
                "dataset_protocol": data.get("dataset_protocol", ""),
                "split": data.get("split", ""),
                "row": "all",
                "precision": metric_value(overall, "metrics/precision(B)", "precision"),
                "recall": metric_value(overall, "metrics/recall(B)", "recall"),
                "map50": metric_value(overall, "metrics/mAP50(B)", "mAP50"),
                "map50_95": metric_value(overall, "metrics/mAP50-95(B)", "mAP50-95"),
                "source_path": str(path.relative_to(PIPELINE_ROOT)),
            }
        )
        for item in data.get("per_class", []):
            rows.append(
                {
                    "task_name": data.get("task_name", path.parent.name),
                    "model_name": data.get("model_name", ""),
                    "scale": data.get("scale", ""),
                    "dataset_protocol": data.get("dataset_protocol", ""),
                    "split": data.get("split", ""),
                    "row": item.get("Class", ""),
                    "precision": metric_value(item, "Box-P", "P"),
                    "recall": metric_value(item, "Box-R", "R"),
                    "map50": metric_value(item, "mAP50"),
                    "map50_95": metric_value(item, "mAP50-95"),
                    "source_path": str(path.relative_to(PIPELINE_ROOT)),
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "final_eval_metrics_flat.csv"
    fields = [
        "task_name",
        "model_name",
        "scale",
        "dataset_protocol",
        "split",
        "row",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "source_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {out_path}")


if __name__ == "__main__":
    main()
