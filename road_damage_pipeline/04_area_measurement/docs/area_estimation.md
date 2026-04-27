# Road Damage Area Estimation / 道路病害面积估计

## Purpose / 目的

This part estimates road-damage area from selected RDD images and GT bounding boxes.

本部分基于筛选出的 RDD 图片和 GT bbox 做病害面积估计。

## Important Boundary / 重要边界

There is no camera calibration, no real focal length, and no ground-truth physical area. Therefore the outputs are estimated areas, not strict real-world measurements.

当前没有相机标定、真实焦距和物理面积 GT。因此输出只能称为估计面积，不能称为严格真实面积。

## Input / 输入

- 10 selected RDD test images
- YOLO-format GT labels
- Converted bbox table: `samples/images/gt_bboxes_from_txt.csv`

## Methods / 方法

### M1: Bbox Geometry Prior

Uses bbox width/height and a fixed pixel-to-metre scale. D00 uses bbox height x 0.8, D10 uses bbox width x 1.2, and D20/D40 use one third of the bbox rectangle area.

使用 bbox 宽高和固定像素到米的比例计算。D00 按 bbox 高度 x 0.8，D10 按 bbox 宽度 x 1.2，D20/D40 按 bbox 矩形面积的 1/3。

### M2: FastSAM mask

Uses the GT bbox crop as the FastSAM input. The text prompt is intentionally coarse: crack-like classes use `crack`, while D40 uses `road damage`.

先裁剪 GT bbox 区域，再把裁剪图传给 FastSAM。文本提示词刻意保持粗粒度：裂缝类使用 `crack`，D40 使用 `road damage`，不把 `D40` 这类标签名当提示词。

### M3: Depth Anything V2 + Bbox Geometry Prior

Uses the full bbox rectangle for Depth Anything V2 depth sampling, converts the bbox rectangle into an approximate depth area with an assumed horizontal FOV, then applies the same class-specific effective-area ratio derived from M1. FastSAM is not used by M3.

使用完整 bbox 矩形区域读取 Depth Anything V2 深度，在假设水平视场角下得到 bbox 矩形的近似深度面积，再乘以 M1 得到的类别有效面积比例。M3 不使用 FastSAM。

### M4: Metric3D + Bbox Geometry Prior

Uses the full bbox rectangle for Metric3D depth sampling, converts the bbox rectangle into an approximate depth area with the same assumed FOV, then applies the same class-specific effective-area ratio.

使用完整 bbox 矩形区域读取 Metric3D 深度，在同一假设视场角下得到 bbox 矩形的近似深度面积，再乘以同类别的有效面积比例。

## Current Interpretation / 当前解释

FastSAM can still overfill a crop when the visual boundary of a crack or pothole is weak. Depth-assisted methods are exploratory because the road surface depth can look nearly uniform without reliable camera calibration.

当裂缝或坑洞边界不明显时，FastSAM 可能把 bbox 内大块区域都涂满。深度辅助方法属于探索性结果，因为没有可靠相机标定时，路面深度可能接近一整片。

## Crop-only Segmentation Visualization / bbox 裁剪分割可视化

For qualitative visualization, the current pipeline uses GT bbox crops only and no PIDNet:

```text
GT bbox -> crop image region -> FastSAM with class-derived damage text prompt
```

This visualization does not show area values. It is used only to inspect whether FastSAM can produce a useful mask inside the annotated damage box.

定性可视化阶段只使用 GT bbox 裁剪图：

```text
GT bbox -> 裁剪 bbox 区域 -> FastSAM 使用类别文本提示词分割病害
```

这组图不展示面积数值，只用于检查 FastSAM 在病害框内部是否能产生可信 mask。

Evidence files / 证据文件：

- Crop visual boards: `assets/area_segmentation_quality_crop/boards/`
- Original bbox panels: `assets/area_segmentation_quality_crop/01_original_bbox/`
- Crop FastSAM prompt masks: `assets/area_segmentation_quality_crop/03_crop_fastsam_prompt_all/`
- Crop FastSAM selected success/failure: `assets/area_segmentation_quality_crop/04_crop_fastsam_prompt_success/`, `assets/area_segmentation_quality_crop/05_crop_fastsam_prompt_failed/`

## Evidence / 证据文件

- Long-format results: `assets/area/four_method_area_results_long.csv`
- Wide-format results: `assets/area/four_method_area_results_wide.csv`
- Class summary: `assets/area/four_method_area_summary_by_class.csv`
- Visual table: `assets/area/visuals/four_method_summary_table.png`
- Per-image boards: `assets/area/visuals/four_method_boards/`
