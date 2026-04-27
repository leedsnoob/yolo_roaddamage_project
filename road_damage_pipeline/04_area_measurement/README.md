# 04 Area Measurement / 病害面积估计

This module contains area-estimation evidence based on selected RDD images and GT bboxes.

本模块保存基于 RDD 选图和 GT bbox 的病害面积估计证据。

## Contents / 内容

- `assets/area/`: four-method area result tables and visual boards.
- `assets/area_segmentation_quality_crop/`: FastSAM crop diagnostics kept as exploration evidence.
- `samples/images/`: selected images, labels, and converted bbox CSV.
- `scripts/build_area_pipeline_v1.py`: combines M1-M4 results into paper-ready tables and boards.
- `scripts/build_crop_segmentation_quality.py`: regenerates FastSAM crop diagnostic visuals.

## Methods / 方法

- `M1`: senior empirical bbox rule with fixed pixel scale.
- `M2`: FastSAM mask on bbox crop, kept as segmentation exploration.
- `M3`: Depth Anything V2 depth inside bbox + senior empirical bbox ratio.
- `M4`: Metric3D depth inside bbox + senior empirical bbox ratio.

The formal report demo passes `M1`, `M3`, and `M4` to Qwen. `M2` is kept as evidence that FastSAM was explored but is not used as the formal report area input.

正式报告 demo 会把 `M1`、`M3`、`M4` 传给 Qwen。`M2` 保留为 FastSAM 探索证据，不作为正式报告面积输入。

## Usage / 使用

Rebuild packaged area evidence:

重新生成打包面积证据：

```bash
python road_damage_pipeline/04_area_measurement/scripts/build_area_pipeline_v1.py
```

Rebuild FastSAM crop diagnostics:

重新生成 FastSAM 裁剪诊断图：

```bash
python road_damage_pipeline/04_area_measurement/scripts/build_crop_segmentation_quality.py
```

## Boundary / 边界

The current default scale is `0.01 m/px`. No lane-line calibration is applied in the packaged result. All numbers are estimated areas, not physical ground truth.

当前默认比例是 `0.01 m/px`。打包结果还没有引入车道线逐图标定。所有数值都是估计面积，不是真实物理 GT。

## Dependency note / 依赖说明

The packaged FastSAM crop diagnostics are already generated. Re-running `scripts/build_crop_segmentation_quality.py` with text prompts requires the CLIP dependency used by Ultralytics FastSAM text prompting.

当前 FastSAM crop 诊断图已经生成。若要重新运行 `scripts/build_crop_segmentation_quality.py` 并使用文本提示词，需要安装 Ultralytics FastSAM 文本提示所需的 CLIP 依赖。
