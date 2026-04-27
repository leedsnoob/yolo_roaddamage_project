# 03 Video Deduplication / 视频去重

This module is separate from image-level detection. It converts repeated frame-level detections into event-level counts.

本模块和图像检测分开维护，目标是把视频连续帧里的重复检测转换成事件级统计。

## Contents / 内容

- `scripts/infer_video.py`: video inference with `ByteTrack`, `BoT-SORT`, and `DeepOC-SORT-lite` backends.
- `scripts/build_dedup_visuals.py`: builds tracker comparison visuals from dense detection frames.
- `pipeline_core/`: tracker adapter and event-merging utilities.
- `samples/videos/3_dense_130_190.mp4`: dense 60-second demo clip.
- `assets/video_dedup/`: tracker comparison CSVs, timelines, keyframes, and side-by-side boards.

## Visualization / 可视化

The side-by-side boards are no longer fixed at `10/30/50s`. They are selected from high-density detection frames with enforced temporal spacing. See:

对比图不再固定为 `10/30/50s`，而是从检测密集且间隔足够远的帧中自动选择。查看：

```text
assets/video_dedup/dedup_visual_boards.csv
assets/video_dedup/dedup_tracker_comparison_*.jpg
assets/video_dedup/keyframes/
```

## Boundary / 边界

This module solves cross-frame duplicate counting. It does not solve duplicate boxes within a single image.

本模块解决跨帧重复计数，不解决单张图内部多个重复框的问题。

## Usage / 使用

Run the 60-second demo clip with ByteTrack:

使用 ByteTrack 运行 60 秒 demo 视频：

```bash
python road_damage_pipeline/03_video_dedup/scripts/infer_video.py \
  --video road_damage_pipeline/03_video_dedup/samples/videos/3_dense_130_190.mp4 \
  --weights road_damage_pipeline/03_video_dedup/weights/yolo11s_original_nondrone_noempty_best.pt \
  --repo-root ultralytics_yolo11_final \
  --tracker-backend bytetrack \
  --dedup-mode track_only \
  --save-video
```

Build tracker comparison visuals:

生成 tracker 对比可视化：

```bash
python road_damage_pipeline/03_video_dedup/scripts/build_dedup_visuals.py
```

The report module uses the video summary, detections, track events, and three high-density representative frames from this module.

报告模块读取本模块的视频 summary、detections、track events，并选取 3 张病害密集代表帧。
