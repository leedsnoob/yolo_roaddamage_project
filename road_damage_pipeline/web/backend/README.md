# Backend / 后端

FastAPI backend for the road damage workbench.

道路病害工作台的 FastAPI 后端。

## Responsibility / 职责

- Accept exactly one uploaded image or video.
- Create an isolated job directory.
- Run existing pipeline scripts in a configured Python environment.
- Expose status and artifacts to the frontend.
- Parse machine-readable progress events from pipeline subprocesses.

- 接收单个图片或视频。
- 创建隔离的 job 目录。
- 在指定 Python 环境中运行现有 pipeline 脚本。
- 向前端暴露状态和产物。
- 解析 pipeline 子进程输出的机器可读进度事件。

## Endpoints / 接口

- `POST /api/jobs`: upload one file and create a job.
- `GET /api/jobs/{job_id}`: read job status.
- `GET /api/jobs/{job_id}/artifacts`: list generated artifacts.
- `/artifacts/{job_id}/...`: serve generated files.

`POST /api/jobs` accepts `report_language=zh|en`. The backend stores it in the job options and forwards it to the report-generation script.

`POST /api/jobs` 支持 `report_language=zh|en`。后端会把它保存到任务配置，并传给报告生成脚本。

`GET /api/jobs/{job_id}` returns `progress_percent`, `active_step`, and per-step messages. These fields are the frontend progress source during long image/video inference jobs.

`GET /api/jobs/{job_id}` 会返回 `progress_percent`、`active_step` 和每一步的说明信息。长时间图片/视频推理时，前端进度条直接读取这些字段。

Long-running scripts emit `PIPELINE_EVENT {...}` lines while processing frames, loading depth resources, writing area evidence, or calling Qwen. The backend consumes these events and updates the job state; it does not rely on a frontend-only timer.

长任务脚本会在处理视频帧、加载深度模型、写入面积证据或调用 Qwen 时输出 `PIPELINE_EVENT {...}`。后端读取这些事件并更新任务状态，不依赖前端固定计时动画。

## Environment / 环境

The backend itself is lightweight, but the job runner needs the pipeline dependencies: `opencv-python`, `torch`, `ultralytics`, `scipy`, `filterpy`, `lap`, `pandas`, `fastapi`, `uvicorn`, and `python-multipart`.

后端本身较轻，但任务 runner 需要完整 pipeline 依赖。

Recommended local command:

```bash
ROAD_DAMAGE_PIPELINE_PYTHON=/Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/python \
  /Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/uvicorn \
  road_damage_pipeline.web.backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

For real Qwen reports, start the backend with `SILICONFLOW_API_KEY` in the shell or place a local `apikey.txt` at the project root before startup. The backend only reports key availability and source type; it does not expose the key value.

如果需要真实 Qwen 报告，需要在启动后端前设置 `SILICONFLOW_API_KEY`，或把本地 `apikey.txt` 放在项目根目录。后端只返回 key 是否可用和来源类型，不返回 key 内容。

## Tests / 测试

```bash
/Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/python \
  -m pytest road_damage_pipeline/web/backend/tests/test_job_api.py -q
```
