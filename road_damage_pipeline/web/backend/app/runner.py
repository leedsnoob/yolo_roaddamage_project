from __future__ import annotations

import json
import os
import shutil
import subprocess
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
        job.mark_step("segmentation", "running", "Linking packaged PIDNet/FastSAM exploration visuals")
        assets = self.settings.pipeline_root / "01_segmentation" / "assets"
        target = job.output_dir / "segmentation_exploration"
        if target.exists():
            shutil.rmtree(target)
        if assets.exists():
            shutil.copytree(assets, target)
            job.mark_step("segmentation", "done", "Packaged exploration visuals attached")
        else:
            job.mark_step("segmentation", "skipped", "Segmentation exploration assets not found")

    def _run_image_job(self, job: JobState) -> None:
        image_dir = job.output_dir / "input_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        normalized_image = image_dir / f"{job.input_path.stem}.jpg"
        self._normalize_image(job.input_path, normalized_image)

        job.mark_step("detection", "running", "Running image detection, area estimation and report evidence build")
        job.mark_step("dedup", "skipped", "Image job does not require video deduplication")
        job.mark_step("area", "running", "Waiting for live M1/M3/M4 area outputs")
        job.mark_step("report", "running" if job.options.call_api else "skipped", "Report API enabled" if job.options.call_api else "API disabled; request preview only")

        command = self._report_command(job, mode="image") + [
            "--image-dir",
            str(image_dir),
            "--max-images",
            "1",
        ]
        result = self._run_command(command, job.output_dir)
        self._update_from_report_output(job, job.output_dir / "image_demo", result)

    def _run_video_job(self, job: JobState) -> None:
        job.mark_step("detection", "running", "Running video detection and tracker")
        job.mark_step("dedup", "running", f"Running {job.options.tracker_backend} track-id deduplication")
        job.mark_step("area", "running", "Estimating area for representative events")
        job.mark_step("report", "running" if job.options.call_api else "skipped", "Report API enabled" if job.options.call_api else "API disabled; request preview only")

        command = self._report_command(job, mode="video") + [
            "--video",
            str(job.input_path),
            "--video-results-dir",
            str(job.output_dir / "force_live_video_results"),
            "--representative-frames",
            "3",
        ]
        result = self._run_command(command, job.output_dir)
        self._update_from_report_output(job, job.output_dir / "video_demo", result)

    def _report_command(self, job: JobState, mode: str) -> list[str]:
        call_api_enabled = job.options.call_api and bool(os.getenv("SILICONFLOW_API_KEY"))
        if job.options.call_api and not call_api_enabled:
            job.summary["api_warning"] = "SILICONFLOW_API_KEY is not set; generated request preview only."
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
            "--max-visual-evidence",
            "5",
        ]
        if call_api_enabled:
            command.append("--call-api")
        return command

    def _run_command(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        result = subprocess.run(command, cwd=str(self.settings.pipeline_root.parent), env=env, text=True, capture_output=True, check=False)
        log_path = cwd / "pipeline_subprocess.log"
        log_path.write_text(
            "COMMAND:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise PipelineRunError(f"Pipeline command failed with code {result.returncode}. See {log_path}")
        return result

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
