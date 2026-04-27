#!/usr/bin/env python3
"""Run explicit val/test evaluation and export a paper-friendly metrics_summary.json."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PIPELINE_ROOT.parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        try:
            _ = torch.ones(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a road-damage detector.")
    parser.add_argument("--config", type=Path, default=PIPELINE_ROOT / "configs" / "eval_default.yaml")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name", type=str)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def merged_config(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        cfg[key] = value
    return cfg


def metric_value(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return ""


def per_class_rows(results) -> list[dict]:
    box = getattr(results, "box", None)
    names = getattr(results, "names", {}) or {}
    if not box:
        return []
    class_indices = getattr(box, "ap_class_index", [])
    rows = []
    for pos, class_idx in enumerate(class_indices):
        class_idx = int(class_idx)
        rows.append(
            {
                "Class": names.get(class_idx, str(class_idx)),
                "Box-P": round(float(box.p[pos]), 6) if len(box.p) > pos else "",
                "Box-R": round(float(box.r[pos]), 6) if len(box.r) > pos else "",
                "mAP50": round(float(box.ap50[pos]), 6) if len(box.ap50) > pos else "",
                "mAP50-95": round(float(box.ap[pos]), 6) if len(box.ap) > pos else "",
            }
        )
    return rows


def main() -> None:
    cfg = merged_config(parse_args())
    repo_root = resolve_path(cfg.get("repo_root") or "ultralytics_yolo11_final")
    weights = resolve_path(cfg.get("weights"))
    data = resolve_path(cfg.get("data"))
    project = resolve_path(cfg.get("project") or PIPELINE_ROOT / "outputs" / "eval")
    split = str(cfg.get("split", "test"))
    name = str(cfg.get("name") or f"{weights.stem}_{split}") if weights else f"eval_{split}"
    device = resolve_device(str(cfg.get("device", "auto")))

    if not repo_root or not repo_root.exists():
        raise FileNotFoundError(f"YOLO repo root not found: {repo_root}")
    if not weights or not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data or not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ultralytics import YOLO

    start = time.time()
    model = YOLO(str(weights))
    results = model.val(
        data=str(data),
        split=split,
        imgsz=int(cfg.get("imgsz", 832)),
        conf=float(cfg.get("conf", 0.001)),
        iou=float(cfg.get("iou", 0.7)),
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        save_json=bool(cfg.get("save_json", cfg.get("save-json", False))),
    )
    runtime = round(time.time() - start, 3)
    save_dir = Path(getattr(results, "save_dir", project / name))
    overall = getattr(results, "results_dict", {}) or {}
    summary = {
        "task_name": name,
        "dataset_protocol": data.stem,
        "split": split,
        "weight_path": str(weights),
        "save_dir": str(save_dir),
        "overall": {
            "metrics/precision(B)": metric_value(overall, "metrics/precision(B)", "precision"),
            "metrics/recall(B)": metric_value(overall, "metrics/recall(B)", "recall"),
            "metrics/mAP50(B)": metric_value(overall, "metrics/mAP50(B)", "mAP50"),
            "metrics/mAP50-95(B)": metric_value(overall, "metrics/mAP50-95(B)", "mAP50-95"),
            "fitness": metric_value(overall, "fitness"),
        },
        "per_class": per_class_rows(results),
        "speed": getattr(results, "speed", {}),
        "runtime_seconds": runtime,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print(f"metrics_summary: {save_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
