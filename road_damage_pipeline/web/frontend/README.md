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
- Show major visual artifacts first, then list all generated files.
- Show report markdown when `report.md` is available.

- 一次上传一个图片或视频。
- 提交前显示本地预览。
- 展示上传、分割探索、检测、视频去重、面积计算、报告生成状态。
- 优先展示主要可视化产物，再列出全部文件。
- 如果存在 `report.md`，直接显示报告内容。
