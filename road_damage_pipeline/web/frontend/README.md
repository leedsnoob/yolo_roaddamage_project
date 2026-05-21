# Frontend / 前端

React + TypeScript + Vite frontend for the road damage workbench.

道路病害工作台的 React + TypeScript + Vite 前端。

## Start / 启动

```bash
cd road_damage_pipeline/web/frontend
npm install
npm run dev
```

The dev server proxies `/api` and `/artifacts` to `http://127.0.0.1:8000`.

开发服务器会把 `/api` 和 `/artifacts` 代理到 `http://127.0.0.1:8000`。

## Build / 构建

```bash
npm run build
```

## UI behavior / 界面行为

- Upload one image or video.
- Show local preview before submitting.
- Display step states for upload, segmentation exploration, detection, video deduplication, area measurement, and report generation.
- Display a job progress bar and current active step while the backend is still running.
- Display per-step progress bars fed by backend pipeline events rather than a fixed timer.
- Show major visual artifacts first, then list all generated files.
- Preview generated images, videos, reports, CSV and JSON files in the main panel; downloading is a separate action.
- For video jobs, prefer the generated annotated video as the main artifact when it is available.
- Show report markdown when `report.md` is available.
- Switch between Chinese and English. The selected language is sent to the backend as `report_language`.
- Render generated Markdown reports as readable report pages and provide a print/save-as-PDF action.
- Add a cache-busting version to artifact URLs so refreshed reports and images are not confused with browser-cached copies.

- 一次上传一个图片或视频。
- 提交前显示本地预览。
- 展示上传、分割探索、检测、视频去重、面积计算、报告生成状态。
- 后端运行时展示任务进度条和当前步骤。
- 展示由后端 pipeline 事件驱动的每一步进度条，而不是固定计时动画。
- 优先展示主要可视化产物，再列出全部文件。
- 在主面板预览图片、视频、报告、CSV 和 JSON；下载是单独动作。
- 视频任务完成后，优先把生成的带框视频作为主产物展示。
- 如果存在 `report.md`，直接显示报告内容。
- 支持中文和英文切换，并把选择结果作为 `report_language` 传给后端。
- 将生成的 Markdown 报告渲染为易读报告页面，并提供打印/另存 PDF 操作。
- 给产物 URL 添加版本参数，避免浏览器缓存让新报告或新图片看起来没有更新。
