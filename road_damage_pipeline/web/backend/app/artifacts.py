from __future__ import annotations

from pathlib import Path
from typing import Any


VISIBLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".csv", ".json", ".md", ".txt"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov"}


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix == ".md":
        return "report"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    return "file"


def list_artifacts(job_id: str, output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VISIBLE_SUFFIXES:
            continue
        rel = path.relative_to(output_dir)
        artifacts.append(
            {
                "name": path.name,
                "relative_path": str(rel),
                "kind": artifact_kind(path),
                "size_bytes": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
                "url": f"/artifacts/{job_id}/{rel.as_posix()}",
            }
        )
    return artifacts
