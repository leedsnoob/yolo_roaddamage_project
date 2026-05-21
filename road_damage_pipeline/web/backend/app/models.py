from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


StepStatus = Literal["pending", "running", "done", "failed", "skipped"]
JobStatus = Literal["queued", "running", "completed", "failed"]
FileType = Literal["image", "video"]
ReportLanguage = Literal["zh", "en"]


STEP_ORDER = ["upload", "segmentation", "detection", "dedup", "area", "report"]
STEP_PROGRESS_WEIGHTS = {
    "upload": 8,
    "segmentation": 10,
    "detection": 30,
    "dedup": 16,
    "area": 22,
    "report": 14,
}


@dataclass
class StepState:
    status: StepStatus = "pending"
    message: str = ""
    progress: int = 0
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class JobOptions:
    run_segmentation: bool = False
    call_api: bool = False
    report_language: ReportLanguage = "zh"
    conf: float = 0.25
    iou: float = 0.50
    imgsz: int = 832
    device: str = "auto"
    tracker_backend: str = "bytetrack"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_segmentation": self.run_segmentation,
            "call_api": self.call_api,
            "report_language": self.report_language,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "tracker_backend": self.tracker_backend,
        }


@dataclass
class JobState:
    job_id: str
    file_name: str
    file_type: FileType
    input_path: Path
    output_dir: Path
    options: JobOptions = field(default_factory=JobOptions)
    status: JobStatus = "queued"
    error: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, StepState] = field(default_factory=lambda: {step: StepState() for step in STEP_ORDER})
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def mark_step(self, step: str, status: StepStatus, message: str = "", progress: int | None = None) -> None:
        state = self.steps.setdefault(step, StepState())
        if status == "running" and state.started_at is None:
            state.started_at = time.time()
        if status in {"done", "failed", "skipped"}:
            if state.started_at is None:
                state.started_at = time.time()
            state.finished_at = time.time()
        state.status = status
        state.message = message
        if progress is not None:
            state.progress = max(0, min(100, int(progress)))
        elif status in {"done", "skipped"}:
            state.progress = 100
        elif status == "pending":
            state.progress = 0
        elif status == "running" and state.progress <= 0:
            state.progress = 1
        self.updated_at = time.time()

    def update_step_progress(self, step: str, progress: int, message: str | None = None) -> None:
        state = self.steps.setdefault(step, StepState())
        if state.status not in {"running", "done", "failed", "skipped"}:
            state.status = "running"
        if state.started_at is None:
            state.started_at = time.time()
        state.progress = max(state.progress, max(0, min(99, int(progress))))
        if message:
            state.message = message
        self.updated_at = time.time()

    def active_step(self) -> str:
        for step in STEP_ORDER:
            if self.steps.get(step, StepState()).status == "running":
                return step
        for step in STEP_ORDER:
            if self.steps.get(step, StepState()).status == "failed":
                return step
        for step in STEP_ORDER:
            if self.steps.get(step, StepState()).status == "pending":
                return step
        return "complete"

    def progress_percent(self) -> int:
        if self.status == "completed":
            return 100
        total = sum(STEP_PROGRESS_WEIGHTS.values())
        score = 0.0
        for step in STEP_ORDER:
            weight = STEP_PROGRESS_WEIGHTS[step]
            status = self.steps.get(step, StepState()).status
            if status in {"done", "skipped"}:
                score += weight
            elif status == "running":
                state = self.steps.get(step, StepState())
                score += weight * max(1, min(99, state.progress)) / 100.0
            elif status == "failed":
                score += weight * 0.35
        return max(0, min(99, round((score / total) * 100)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "status": self.status,
            "error": self.error,
            "summary": self.summary,
            "steps": {name: state.to_dict() for name, state in self.steps.items()},
            "options": self.options.to_dict(),
            "active_step": self.active_step(),
            "progress_percent": self.progress_percent(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
