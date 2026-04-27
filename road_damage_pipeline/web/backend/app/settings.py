from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def default_pipeline_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_pipeline_python() -> str:
    env_python = os.getenv("ROAD_DAMAGE_PIPELINE_PYTHON")
    if env_python:
        return env_python
    pipeline_root = default_pipeline_root()
    sibling_conda = pipeline_root.parent.parent / "video_damage_analytics" / ".conda-yolo311" / "bin" / "python"
    if sibling_conda.exists():
        return str(sibling_conda)
    return sys.executable


@dataclass
class AppSettings:
    pipeline_root: Path = default_pipeline_root()
    output_root: Path = default_pipeline_root() / "outputs" / "ui_jobs"
    pipeline_python: str = default_pipeline_python()
    max_image_bytes: int = 25 * 1024 * 1024
    max_video_bytes: int = 500 * 1024 * 1024
    max_video_seconds: int = 10 * 60
    default_device: str = "auto"
    default_imgsz: int = 832
    default_conf: float = 0.25
    default_iou: float = 0.50
    default_tracker_backend: str = "bytetrack"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov"}
