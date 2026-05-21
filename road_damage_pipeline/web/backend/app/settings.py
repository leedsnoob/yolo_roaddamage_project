from __future__ import annotations

import os
import re
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


def candidate_api_key_paths(pipeline_root: Path) -> list[Path]:
    """Return local-only API key file candidates without exposing their contents."""
    candidates: list[Path] = []
    env_path = os.getenv("SILICONFLOW_API_KEY_FILE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            pipeline_root / "apikey.txt",
            pipeline_root.parent / "apikey.txt",
            pipeline_root.parent.parent / "apikey.txt",
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_siliconflow_api_key(settings: "AppSettings") -> str:
    """Load SILICONFLOW_API_KEY from environment or local apikey.txt.

    The value is only written to process environment for subprocess inheritance.
    It is never returned, logged, or exposed through the API.
    """
    if os.getenv("SILICONFLOW_API_KEY"):
        settings.siliconflow_api_source = "environment"
        return settings.siliconflow_api_source

    for path in candidate_api_key_paths(settings.pipeline_root):
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"sk-[A-Za-z0-9]+", content)
        if not match:
            settings.siliconflow_api_source = f"invalid_file:{path.name}"
            return settings.siliconflow_api_source
        os.environ["SILICONFLOW_API_KEY"] = match.group(0)
        settings.siliconflow_api_source = f"file:{path.name}"
        return settings.siliconflow_api_source

    settings.siliconflow_api_source = "missing"
    return settings.siliconflow_api_source


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
    siliconflow_api_source: str = "not_checked"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov"}
