from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import JobState
from app.settings import AppSettings


class FakeRunner:
    def __init__(self, *, no_detections: bool = False) -> None:
        self.no_detections = no_detections

    def run(self, job: JobState) -> None:
        job.mark_step("upload", "done", "File stored")
        job.mark_step("segmentation", "skipped", "Exploration module disabled")
        job.mark_step("detection", "running", "Fake detector")
        if self.no_detections:
            job.summary["total_detections"] = 0
            job.mark_step("detection", "done", "No damage detected")
            job.mark_step("area", "skipped", "No detections")
            job.mark_step("report", "skipped", "No detections")
            return

        artifact = job.output_dir / "image_demo" / "report.md"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("# Fake report\n", encoding="utf-8")
        (job.output_dir / "image_demo" / "report_input.json").write_text(
            json.dumps({"damage_summary": {"total_detections": 1}}),
            encoding="utf-8",
        )
        job.summary["total_detections"] = 1
        job.mark_step("detection", "done", "1 detection")
        job.mark_step("area", "done", "Fake area")
        job.mark_step("report", "done", "Fake report")


def make_client(tmp_path: Path, runner: FakeRunner | None = None) -> TestClient:
    settings = AppSettings(
        pipeline_root=Path.cwd() / "road_damage_pipeline",
        output_root=tmp_path / "ui_jobs",
        max_image_bytes=1024 * 1024,
        max_video_bytes=1024 * 1024,
    )
    app = create_app(settings=settings, runner=runner or FakeRunner())
    return TestClient(app)


def test_upload_image_creates_completed_job_with_artifacts(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/jobs",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
        data={"run_segmentation": "false", "call_api": "false", "report_language": "en"},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["file_type"] == "image"
    assert status["status"] == "completed"
    assert status["steps"]["detection"]["status"] == "done"
    assert status["steps"]["detection"]["progress"] == 100
    assert status["steps"]["report"]["status"] == "done"
    assert status["progress_percent"] == 100
    assert status["options"]["report_language"] == "en"

    artifacts = client.get(f"/api/jobs/{job_id}/artifacts").json()
    paths = {item["name"]: item for item in artifacts["artifacts"]}
    assert "report.md" in paths
    assert paths["report.md"]["url"].startswith(f"/artifacts/{job_id}/")
    assert isinstance(paths["report.md"]["modified_at"], float)


def test_upload_preserves_sanitized_original_filename(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/jobs",
        files={"file": ("Japan 000050.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    artifacts = client.get(f"/api/jobs/{job_id}/artifacts").json()["artifacts"]
    relative_paths = {item["relative_path"] for item in artifacts}
    assert "upload/Japan_000050.jpg" in relative_paths
    assert "upload/uploaded.jpg" not in relative_paths


def test_upload_rejects_unsupported_file_type(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/jobs",
        files={"file": ("notes.txt", b"bad", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_no_detection_job_skips_area_and_report(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeRunner(no_detections=True))
    response = client.post(
        "/api/jobs",
        files={"file": ("empty.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = client.get(f"/api/jobs/{job_id}").json()

    assert status["status"] == "completed"
    assert status["summary"]["total_detections"] == 0
    assert status["steps"]["area"]["status"] == "skipped"
    assert status["steps"]["area"]["progress"] == 100
    assert status["steps"]["report"]["status"] == "skipped"
