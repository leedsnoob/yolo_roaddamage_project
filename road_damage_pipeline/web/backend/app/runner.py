from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .models import JobState
from .settings import AppSettings


class PipelineRunError(RuntimeError):
    pass


class PipelineRunner:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def run(self, job: JobState) -> None:
        job.mark_step("upload", "done", "File stored")
        if job.options.run_segmentation:
            self._attach_segmentation_exploration(job)
        else:
            job.mark_step("segmentation", "skipped", "Exploration module disabled")

        if job.file_type == "image":
            self._run_image_job(job)
        else:
            self._run_video_job(job)

    def _attach_segmentation_exploration(self, job: JobState) -> None:
        if job.file_type != "image":
            job.mark_step("segmentation", "skipped", "Live segmentation exploration is image-only in v1")
            return

        job.mark_step("segmentation", "running", "Running live PIDNet road segmentation for uploaded image", progress=5)
        target = job.output_dir / "segmentation_exploration"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        command = [
            self.settings.pipeline_python,
            str(self.settings.pipeline_root / "01_segmentation" / "scripts" / "segment_uploaded_image.py"),
            "--image",
            str(job.input_path),
            "--output-dir",
            str(target),
            "--device",
            job.options.device,
        ]
        try:
            result = self._run_command(command, job.output_dir)
        except PipelineRunError as exc:
            job.mark_step("segmentation", "skipped", f"Live segmentation unavailable: {exc}")
            return

        summary_path = target / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
            if summary.get("status") == "done":
                job.mark_step("segmentation", "done", "Live PIDNet segmentation generated for uploaded image")
                return
            job.mark_step("segmentation", "skipped", str(summary.get("reason", "Segmentation resources unavailable")))
            return
        job.mark_step("segmentation", "skipped", result.stdout[-300:] or "Segmentation output missing")

    def _run_image_job(self, job: JobState) -> None:
        image_dir = job.output_dir / "input_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        normalized_image = image_dir / f"{job.input_path.stem}.jpg"
        self._normalize_image(job.input_path, normalized_image)

        job.mark_step("detection", "running", "Running YOLO image detection and building report evidence", progress=5)
        job.mark_step("dedup", "skipped", "Image job does not require video deduplication")
        job.mark_step("area", "running", "Waiting for live bbox-geometry and depth-assisted area outputs", progress=1)
        job.mark_step(
            "report",
            "running" if job.options.call_api else "skipped",
            "Qwen API requested; waiting for structured evidence"
            if job.options.call_api
            else "Qwen API call is disabled for this job; only local evidence and request preview will be written",
            progress=1 if job.options.call_api else 100,
        )

        command = self._report_command(job, mode="image") + [
            "--image-dir",
            str(image_dir),
            "--max-images",
            "1",
        ]
        result = self._run_command(command, job.output_dir, job=job, demo_dir=job.output_dir / "image_demo")
        self._update_from_report_output(job, job.output_dir / "image_demo", result)

    def _run_video_job(self, job: JobState) -> None:
        job.mark_step("detection", "running", "Running YOLO video detection", progress=3)
        job.mark_step("dedup", "running", f"Running {job.options.tracker_backend} track-id deduplication", progress=1)
        job.mark_step("area", "running", "Estimating area for representative events", progress=1)
        job.mark_step(
            "report",
            "running" if job.options.call_api else "skipped",
            "Qwen API requested; waiting for video evidence"
            if job.options.call_api
            else "Qwen API call is disabled for this job; only local evidence and request preview will be written",
            progress=1 if job.options.call_api else 100,
        )

        command = self._report_command(job, mode="video") + [
            "--video",
            str(job.input_path),
            "--video-results-dir",
            str(job.output_dir / "force_live_video_results"),
            "--representative-frames",
            "3",
        ]
        result = self._run_command(command, job.output_dir, job=job, demo_dir=job.output_dir / "video_demo")
        self._update_from_report_output(job, job.output_dir / "video_demo", result)

    def _report_command(self, job: JobState, mode: str) -> list[str]:
        call_api_enabled = job.options.call_api and bool(os.getenv("SILICONFLOW_API_KEY"))
        if job.options.call_api and not call_api_enabled:
            job.summary["api_warning"] = (
                "Real Qwen report was requested, but the backend process cannot see SILICONFLOW_API_KEY. "
                "Set SILICONFLOW_API_KEY or provide a local apikey.txt before starting the backend, then submit a new job. "
                "This job will only generate qwen_request_preview.json."
            )
        command = [
            self.settings.pipeline_python,
            str(self.settings.pipeline_root / "05_report_generation" / "scripts" / "report_pipeline.py"),
            "--mode",
            mode,
            "--output-root",
            str(job.output_dir),
            "--repo-root",
            str(self.settings.pipeline_root.parent),
            "--device",
            job.options.device,
            "--imgsz",
            str(job.options.imgsz),
            "--conf",
            str(job.options.conf),
            "--iou",
            str(job.options.iou),
            "--tracker-backend",
            job.options.tracker_backend,
            "--report-language",
            job.options.report_language,
            "--max-visual-evidence",
            "5",
        ]
        if call_api_enabled:
            command.append("--call-api")
        return command

    def _run_command(
        self,
        command: list[str],
        cwd: Path,
        job: JobState | None = None,
        demo_dir: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command,
            cwd=str(self.settings.pipeline_root.parent),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        stdout_parts: list[str] = []
        last_refresh = time.time()
        while True:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                stdout_parts.append(line)
                if job is not None:
                    self._handle_pipeline_event_line(job, line)
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.1)
            if job is not None and demo_dir is not None and time.time() - last_refresh >= 1.0:
                self._refresh_partial_progress(job, demo_dir)
                last_refresh = time.time()
        stdout = "".join(stdout_parts)
        result = subprocess.CompletedProcess(command, process.returncode, stdout, "")
        log_path = cwd / "pipeline_subprocess.log"
        log_path.write_text(
            "COMMAND:\n" + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise PipelineRunError(f"Pipeline command failed with code {result.returncode}. See {log_path}")
        return result

    def _handle_pipeline_event_line(self, job: JobState, line: str) -> None:
        prefix = "PIPELINE_EVENT "
        if prefix not in line:
            return
        payload_text = line.split(prefix, 1)[1].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return
        step = str(payload.get("step", ""))
        if step not in job.steps:
            return
        progress = int(payload.get("progress", job.steps[step].progress))
        message = str(payload.get("message", ""))
        status = payload.get("status")
        if status in {"pending", "running", "done", "failed", "skipped"}:
            job.mark_step(step, status, message or job.steps[step].message, progress=progress)
        else:
            job.update_step_progress(step, progress, message or None)

    def _refresh_partial_progress(self, job: JobState, demo_dir: Path) -> None:
        """Update UI-visible step state while the report subprocess is still running."""
        report_input_path = demo_dir / "report_input.json"
        if report_input_path.exists():
            try:
                report_input = json.loads(report_input_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report_input = {}
            summary = report_input.get("damage_summary", {})
            if isinstance(summary, dict):
                job.summary.update(summary)
                total = int(summary.get("total_detections", 0) or 0)
                if total == 0 and job.steps.get("detection") and job.steps["detection"].status == "running":
                    job.mark_step("detection", "done", "No damage detected")
                    job.mark_step("area", "skipped", "No detections")
                    job.mark_step("report", "skipped", "No detections")
                    return
                if total > 0 and job.steps.get("detection") and job.steps["detection"].status == "running":
                    job.mark_step("detection", "done", f"{total} detections")
                if job.file_type == "video" and job.steps.get("dedup") and job.steps["dedup"].status == "running":
                    unique = int(summary.get("unique_events", 0) or 0)
                    job.mark_step("dedup", "done", f"{unique} unique events")

        if (demo_dir / "area_estimates.csv").exists() or (demo_dir / "event_area_estimates.csv").exists():
            if job.steps.get("area") and job.steps["area"].status == "running":
                job.mark_step("area", "done", "Area estimates and visual evidence generated")

        if not job.options.call_api and (demo_dir / "qwen_request_preview.json").exists():
            if job.steps.get("report") and job.steps["report"].status == "running":
                job.mark_step("report", "skipped", "API disabled; request preview generated")

    def _update_from_report_output(self, job: JobState, demo_dir: Path, result: subprocess.CompletedProcess[str]) -> None:
        report_input_path = demo_dir / "report_input.json"
        report_input: dict[str, Any] = {}
        if report_input_path.exists():
            report_input = json.loads(report_input_path.read_text(encoding="utf-8"))
        summary = report_input.get("damage_summary", {})
        total = int(summary.get("total_detections", 0) or 0)
        job.summary.update(summary)
        job.summary["report_input"] = str(report_input_path) if report_input_path.exists() else ""
        job.summary["pipeline_stdout_tail"] = result.stdout[-1200:]

        if total == 0:
            job.mark_step("detection", "done", "No damage detected")
            job.mark_step("area", "skipped", "No detections")
            job.mark_step("report", "skipped", "No detections")
            if job.file_type == "video":
                job.mark_step("dedup", "done", "No unique damage events")
            return

        job.mark_step("detection", "done", f"{total} detections")
        if job.file_type == "video":
            unique = int(summary.get("unique_events", 0) or 0)
            job.mark_step("dedup", "done", f"{unique} unique events")
        if (demo_dir / "area_estimates.csv").exists() or (demo_dir / "event_area_estimates.csv").exists():
            job.mark_step("area", "done", "Area estimates and visual evidence generated")
        else:
            job.mark_step("area", "skipped", "Area output not generated")
        report_path = demo_dir / "report.md"
        if job.options.call_api and job.summary.get("api_warning"):
            job.mark_step("report", "skipped", str(job.summary["api_warning"]))
        elif job.options.call_api:
            job.mark_step("report", "done" if report_path.exists() else "failed", "Qwen report generated" if report_path.exists() else "Report missing")
        else:
            job.mark_step("report", "skipped", "API disabled; request preview generated")

    def _normalize_image(self, src: Path, dst: Path) -> None:
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(src, dst)
            return
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise PipelineRunError("PNG upload requires opencv-python in the pipeline environment.") from exc
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise PipelineRunError(f"Failed to decode uploaded image: {src.name}")
        cv2.imwrite(str(dst), image)
