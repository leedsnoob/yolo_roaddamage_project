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


@dataclass
class StepState:
    status: StepStatus = "pending"
    message: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
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

    def mark_step(self, step: str, status: StepStatus, message: str = "") -> None:
        state = self.steps.setdefault(step, StepState())
        if status == "running" and state.started_at is None:
            state.started_at = time.time()
        if status in {"done", "failed", "skipped"}:
            if state.started_at is None:
                state.started_at = time.time()
            state.finished_at = time.time()
        state.status = status
        state.message = message
        self.updated_at = time.time()

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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
