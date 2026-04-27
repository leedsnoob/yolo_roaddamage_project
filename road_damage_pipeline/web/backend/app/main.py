from __future__ import annotations

import shutil
import threading
import uuid
import re
import os
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .artifacts import list_artifacts
from .models import JobOptions, JobState
from .runner import PipelineRunner
from .settings import AppSettings, IMAGE_SUFFIXES, VIDEO_SUFFIXES


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.RLock()

    def add(self, job: JobState) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> JobState:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job

    def run_job(self, job_id: str, runner: Any) -> None:
        job = self.get(job_id)
        job.status = "running"
        try:
            runner.run(job)
            if job.status != "failed":
                job.status = "completed"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            for name, state in job.steps.items():
                if state.status == "running":
                    job.mark_step(name, "failed", job.error)


def classify_upload(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '<none>'}")


def safe_upload_name(filename: str) -> str:
    path = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    if not stem:
        stem = "uploaded"
    return f"{stem}{path.suffix.lower()}"


def validate_video_duration(path: Path, max_seconds: int) -> None:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=500, detail="opencv-python is required for video validation.") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Uploaded video cannot be opened.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise HTTPException(status_code=400, detail="Uploaded video metadata is invalid.")
    duration_s = frames / fps
    if duration_s > max_seconds:
        raise HTTPException(status_code=413, detail=f"Video is longer than the configured {max_seconds // 60} minute limit.")


def create_app(settings: AppSettings | None = None, runner: Any | None = None) -> FastAPI:
    settings = settings or AppSettings()
    settings.output_root.mkdir(parents=True, exist_ok=True)
    store = JobStore()
    runner = runner or PipelineRunner(settings)

    app = FastAPI(title="Road Damage Pipeline UI API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/artifacts", StaticFiles(directory=str(settings.output_root)), name="artifacts")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "pipeline_root": str(settings.pipeline_root),
            "pipeline_python": settings.pipeline_python,
            "siliconflow_api_ready": bool(os.getenv("SILICONFLOW_API_KEY")),
        }

    @app.post("/api/jobs")
    async def create_job(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        run_segmentation: bool = Form(False),
        call_api: bool = Form(False),
        conf: float = Form(0.25),
        iou: float = Form(0.50),
        imgsz: int = Form(832),
        device: str = Form("auto"),
        tracker_backend: str = Form("bytetrack"),
    ) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
        file_type = classify_upload(file.filename)

        job_id = uuid.uuid4().hex[:12]
        job_dir = settings.output_root / job_id
        input_dir = job_dir / "upload"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / safe_upload_name(file.filename)
        with input_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        if input_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if file_type == "image" and input_path.stat().st_size > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="Image is larger than the configured 25MB limit.")
        if file_type == "video":
            if input_path.stat().st_size > settings.max_video_bytes:
                raise HTTPException(status_code=413, detail="Video is larger than the configured 500MB limit.")
            validate_video_duration(input_path, settings.max_video_seconds)

        options = JobOptions(
            run_segmentation=run_segmentation,
            call_api=call_api,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            tracker_backend=tracker_backend,
        )
        job = JobState(
            job_id=job_id,
            file_name=file.filename,
            file_type=file_type,  # type: ignore[arg-type]
            input_path=input_path,
            output_dir=job_dir,
            options=options,
        )
        job.mark_step("upload", "done", "File stored")
        store.add(job)
        background_tasks.add_task(store.run_job, job_id, runner)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return store.get(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/api/jobs/{job_id}/artifacts")
    def get_artifacts(job_id: str) -> dict[str, Any]:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        return {"job_id": job_id, "artifacts": list_artifacts(job_id, job.output_dir)}

    return app


app = create_app()
