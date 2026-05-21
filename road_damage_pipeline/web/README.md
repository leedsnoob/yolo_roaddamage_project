# Road Damage Web UI / 道路病害 Web 工作台

This UI is a local engineering wrapper around the five pipeline modules. It does not duplicate YOLO source code. The backend calls the existing Python pipeline scripts and exposes their artifacts to the React frontend.

这个界面是五个 pipeline 模块的本地工程化封装，不复制 YOLO 源码。后端调用现有 Python pipeline 脚本，再把输出产物暴露给 React 前端。

## Flow / 流程

- Image / 图片：upload -> detection -> area measurement -> report generation.
- Video / 视频：upload -> detection -> video deduplication -> representative-frame area measurement -> report generation.
- Segmentation / 语义分割：exploration-only visual module. For image uploads, the UI runs live PIDNet road segmentation on the uploaded image; it is not part of the formal report chain.
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

Alternatively, for local-only development, put the key in `apikey.txt` at the project root before starting the backend. The backend reads it into memory and only reports whether the key is available; the key value is never returned by the API.

也可以在本地开发时把 key 放到项目根目录的 `apikey.txt`，再启动后端。后端只把它读入进程环境，并只返回“是否可用”和来源类型，不返回 key 内容。

Do not write API keys into source files, README files, JSON outputs, or screenshots. `apikey.txt` and `.env*` are ignored by Git.

不要把 API key 写进源码、README、JSON 输出或截图。`apikey.txt` 和 `.env*` 已加入 Git 忽略。

## Output / 输出

Each UI job writes to:

每个 UI 任务输出到：

```text
road_damage_pipeline/outputs/ui_jobs/<job_id>/
```

The frontend shows detection images, video/keyframe artifacts, area boards, depth maps, CSV/JSON evidence, and generated report markdown.

前端会展示检测图、视频/关键帧、面积 board、深度图、CSV/JSON 证据和报告 Markdown。

Generated artifacts can be selected inside the main panel. Images, annotated videos, reports, CSV files and JSON files are previewed in place; download is a separate action.

生成的产物可以在主面板内选择预览。图片、带框视频、报告、CSV 和 JSON 都会在页面内展示；下载是单独操作。

Reports are stored as Markdown evidence, but the frontend renders them as readable report pages and provides a browser print/save-as-PDF action.

报告文件以 Markdown 作为可追溯证据保存，但前端会渲染为易读报告页面，并提供浏览器打印/另存 PDF。

The job status API returns `progress_percent`, `active_step`, and per-step messages. The frontend polls these fields during inference, so long video jobs show visible progress instead of appearing frozen.

任务状态 API 会返回 `progress_percent`、`active_step` 和每一步的状态说明。前端会在推理期间轮询这些字段，因此视频长任务不会表现为无响应。

Browser logs may show `304 Not Modified` for files under `/artifacts`. This is normal HTTP cache validation, not a pipeline error. The frontend appends a job update version to artifact URLs to reduce stale report/image display after reruns.

浏览器日志中 `/artifacts` 文件可能出现 `304 Not Modified`。这是正常的 HTTP 缓存协商，不是 pipeline 错误。前端会给产物 URL 添加任务更新时间版本，减少重新运行后看到旧图片或旧报告的情况。
