# Road Damage Web UI / 道路病害 Web 工作台

This UI is a local engineering wrapper around the five pipeline modules. It does not duplicate YOLO source code. The backend calls the existing Python pipeline scripts and exposes their artifacts to the React frontend.

这个界面是五个 pipeline 模块的本地工程化封装，不复制 YOLO 源码。后端调用现有 Python pipeline 脚本，再把输出产物暴露给 React 前端。

## Flow / 流程

- Image / 图片：upload -> detection -> area measurement -> report generation.
- Video / 视频：upload -> detection -> video deduplication -> representative-frame area measurement -> report generation.
- Segmentation / 语义分割：exploration-only visual module. It can be attached for display, but it is not part of the formal report chain.
- Language / 语言：the UI supports Chinese and English. The selected language is also passed to report generation, so Qwen reports can be generated in Chinese or English.

## Start / 启动

Backend:

```bash
cd /Users/tomchen/Downloads/source_only_clean_20260401/ultralytics_yolo11_final
ROAD_DAMAGE_PIPELINE_PYTHON=/Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/python \
  /Users/tomchen/Downloads/source_only_clean_20260401/video_damage_analytics/.conda-yolo311/bin/uvicorn \
  road_damage_pipeline.web.backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd road_damage_pipeline/web/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## API key / API 密钥

If Qwen reports are needed, set the API key only in the shell:

如果需要真实 Qwen 报告，只在 shell 中设置：

```bash
export SILICONFLOW_API_KEY="your_key_here"
```

Do not write API keys into source files, README files, or JSON outputs.

不要把 API key 写进源码、README 或 JSON 输出。

## Output / 输出

Each UI job writes to:

每个 UI 任务输出到：

```text
road_damage_pipeline/outputs/ui_jobs/<job_id>/
```

The frontend shows detection images, video/keyframe artifacts, area boards, depth maps, CSV/JSON evidence, and generated report markdown.

前端会展示检测图、视频/关键帧、面积 board、深度图、CSV/JSON 证据和报告 Markdown。
