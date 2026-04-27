# Video Detection and Deduplication / 视频检测与去重

## Purpose / 目的

A road damage appears in multiple video frames. The video pipeline turns frame-level detections into event-level counts.

同一个病害会在视频连续多帧里出现。视频 pipeline 的目标是把逐帧检测转换成事件级统计。

## Default Pipeline / 默认流程

```text
video -> YOLO detector -> tracker -> track_id events -> CSV/JSON/annotated video
```

默认流程：

```text
视频 -> YOLO 检测器 -> 跟踪器 -> track_id 事件 -> CSV/JSON/带框视频
```

## Trackers / 跟踪器

Implemented comparison:

已实现对比：

- ByteTrack
- BoT-SORT
- DeepOC-SORT-lite

The default runnable script uses `ByteTrack` because it is fast and easy to reproduce. The paper-assets folder keeps the comparison result showing that DeepOC-SORT-lite produced fewer fragmentation events on the selected dense clip.

默认脚本使用 `ByteTrack`，因为速度快且复现简单。论文证据中保留三种方法对比，其中 DeepOC-SORT-lite 在所选密集片段上碎片化代理指标更低。

## Important Boundary / 重要边界

This is cross-frame deduplication. It is different from removing duplicate boxes inside one image.

这里做的是跨帧去重，和单帧内重复框去重不是同一个问题。

`WIoU` is a training-time box regression loss. It is not a video deduplication algorithm.

`WIoU` 是训练阶段 bbox 回归损失，不是视频去重算法。

## Demo Clip / 演示视频

- Source: `3.mp4`
- Selected segment: `130.13s - 190.13s`
- Duration: about `60s`
- Local file: `samples/videos/3_dense_130_190.mp4`

## Outputs / 输出

The inference script writes:

推理脚本输出：

- `detections.csv`
- `track_events.csv`
- `summary.json`
- annotated video when `save_video=true`

## Evidence / 证据文件

- Tracker comparison table: `assets/video_dedup/tracker_summary.csv`
- Tracker comparison figure: `assets/video_dedup/tracker_summary.png`
- Side-by-side tracker boards: see `assets/video_dedup/dedup_visual_boards.csv`; boards are selected from dense detection frames with enforced temporal spacing.
- Track duration and hit distribution: `assets/video_dedup/dedup_event_duration_distribution.png`
- Density timeline: `assets/video_dedup/density_timeline.png`
- Event timeline: `assets/video_dedup/event_timeline.png`
- Keyframes: `assets/video_dedup/keyframes/`

## Recommended Visualization / 推荐展示方式

For a thesis or demo page, use two visual levels:

论文或展示页建议用两层可视化：

1. Frame-level qualitative comparison:
   - Use one of the boards listed in `dedup_visual_boards.csv`, for example the high-density frame around `45s`.
   - It places ByteTrack, BoT-SORT and DeepOC-SORT-lite on the same frame.
   - This makes ID assignment and box continuity visible.

2. Event-level statistical visualization:
   - Use `dedup_event_duration_distribution.png`.
   - Longer duration and more hits per event usually mean fewer broken tracks.
   - Combine this with `tracker_summary.csv` to explain fragmentation.

不要只放 `unique_track_events` 这种数字。数字说明总体趋势，横向截图说明“为什么这个 tracker 看起来更稳/更碎”。
