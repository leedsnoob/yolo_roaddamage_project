# Backend / 后端

FastAPI backend for the road damage workbench.

道路病害工作台的 FastAPI 后端。

## Responsibility / 职责

- Accept exactly one uploaded image or video.
- Create an isolated job directory.
- Run existing pipeline scripts in a configured Python environment.
- Expose status and artifacts to the frontend.

- 接收单个图片或视频。
- 创建隔离的 job 目录。
- 在指定 Python 环境中运行现有 pipeline 脚本。
- 向前端暴露状态和产物。

## Endpoints / 接口

- `POST /api/jobs`: upload one file and create a job.
- `GET /api/jobs/{job_id}`: read job status.
- `GET /api/jobs/{job_id}/artifacts`: list generated artifacts.
- `/artifacts/{job_id}/...`: serve generated files.

## Environment / 环境

The backend itself is lightweight, but the job runner needs the pipeline dependencies: `opencv-python`, `torch`, `ultralytics`, `scipy`, `filterpy`, `lap`, `pandas`, `fastapi`, `uvicorn`, and `python-multipart`.

后端本身较轻，但任务 runner 需要完整 pipeline 依赖。

Recommended local command:

```bash
ROAD_DAMAGE_PIPELINE_PYTHON=/Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/python \
  /Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/uvicorn \
  road_damage_pipeline.web.backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

## Tests / 测试

```bash
/Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/python \
  -m pytest road_damage_pipeline/web/backend/tests/test_job_api.py -q
```
