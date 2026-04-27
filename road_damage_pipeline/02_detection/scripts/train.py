#!/usr/bin/env python3
"""Train an RDD road-damage detector with the local YOLO final fork."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PIPELINE_ROOT.parent


def default_repo_root() -> Path:
    if (PACKAGE_ROOT.parent / "ultralytics").exists():
        return PACKAGE_ROOT.parent
    return PACKAGE_ROOT.parent / "ultralytics_yolo11_final"


DEFAULT_REPO_ROOT = default_repo_root()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), DEFAULT_REPO_ROOT, DEFAULT_REPO_ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def resolve_repo_root(value: str | Path | None) -> Path:
    if not value:
        return DEFAULT_REPO_ROOT
    path = resolve_path(value)
    if path and path.exists():
        return path
    if Path(value).name == "ultralytics_yolo11_final" and DEFAULT_REPO_ROOT.exists():
        return DEFAULT_REPO_ROOT
    return path or DEFAULT_REPO_ROOT


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
    parser = argparse.ArgumentParser(description="Train a road-damage detector.")
    parser.add_argument("--config", type=Path, default=PIPELINE_ROOT / "configs" / "train_default.yaml")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--close-mosaic", type=int)
    parser.add_argument("--mosaic", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def merged_config(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        cfg[key.replace("_", "-") if key == "close_mosaic" else key] = value
    return cfg


def main() -> None:
    cfg = merged_config(parse_args())
    repo_root = resolve_repo_root(cfg.get("repo_root"))
    model_path = resolve_path(cfg.get("model"))
    pretrained = resolve_path(cfg.get("pretrained"))
    data = resolve_path(cfg.get("data"))
    project = resolve_path(cfg.get("project") or PIPELINE_ROOT / "outputs" / "train")
    device = resolve_device(str(cfg.get("device", "auto")))

    if not repo_root or not repo_root.exists():
        raise FileNotFoundError(f"YOLO repo root not found: {repo_root}")
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")
    if not data or not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if pretrained and pretrained.exists():
        model = model.load(str(pretrained))

    train_kwargs = {
        "data": str(data),
        "epochs": int(cfg.get("epochs", 100)),
        "batch": int(cfg.get("batch", 16)),
        "imgsz": int(cfg.get("imgsz", 832)),
        "device": device,
        "workers": int(cfg.get("workers", 8)),
        "project": str(project),
        "name": str(cfg.get("name", "train")),
        "optimizer": cfg.get("optimizer", "auto"),
        "lr0": float(cfg.get("lr0", 0.01)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "close_mosaic": int(cfg.get("close_mosaic", cfg.get("close-mosaic", 10))),
        "mosaic": float(cfg.get("mosaic", 1.0)),
        "patience": int(cfg.get("patience", 50)),
        "seed": int(cfg.get("seed", 0)),
    }
    project.mkdir(parents=True, exist_ok=True)
    summary_path = project / f"{train_kwargs['name']}_train_config_resolved.json"
    summary_path.write_text(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "model": str(model_path),
                "pretrained": str(pretrained) if pretrained else "",
                "data": str(data),
                "train_kwargs": train_kwargs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
