# PIDNet / FastSAM Usage Notes

This document records verifiable usage facts from the current project.

本文档只记录当前项目中可以追溯的 PIDNet / FastSAM 使用事实。

## Formal PIDRoad Generation / 正式 PIDRoad 生成流程

| Item | Fact |
| --- | --- |
| Model | `PIDNet-L` |
| Checkpoint | `PIDNet_L_Cityscapes_test.pt` |
| Training source | Cityscapes |
| Segmentation type | Semantic segmentation |
| Road class | Cityscapes `road`, class id `0` |
| Prompt | None |
| Bbox preservation | Yes. RDD GT boxes and padding are forced into the keep mask. |
| Current script | `segmentation_pipeline/pidnet/scripts/prepare_pidnet_road_dataset.py` |
| Archived original script | `thesis_package/archive/root_scripts/prepare_pidnet_road_dataset.py` |

| 项目 | 事实 |
| --- | --- |
| 模型 | `PIDNet-L` |
| 权重 | `PIDNet_L_Cityscapes_test.pt` |
| 训练来源 | Cityscapes |
| 分割类型 | 语义分割 |
| 路面类别 | Cityscapes `road`，类别 id 为 `0` |
| 提示词 | 无 |
| bbox 保留 | 是。RDD GT 框和 padding 会被强制加入 keep mask。 |
| 当前脚本 | `segmentation_pipeline/pidnet/scripts/prepare_pidnet_road_dataset.py` |
| 原始归档脚本 | `thesis_package/archive/root_scripts/prepare_pidnet_road_dataset.py` |

## Key Parameters / 关键参数

```text
pidnet architecture: pidnet-l
num_classes: 19
road_class_id: 0
road_dilate: 15
bbox_pad_px: 8
bbox_pad_ratio: 0.02
exclude_drone: true
```

## PIDNet Processing Logic / PIDNet 处理逻辑

```text
1. Run PIDNet on the full image and predict Cityscapes 19-class semantic labels.
2. Select pixels where pred == 0 as the road mask.
3. Dilate the road mask.
4. Read RDD YOLO-format GT boxes and force these regions into the keep mask.
5. Apply the keep mask to the image and copy labels unchanged.
```

```text
1. 对整张图运行 PIDNet，输出 Cityscapes 19 类语义标签。
2. 取 pred == 0 的像素作为 road mask。
3. 对 road mask 做膨胀。
4. 读取 RDD YOLO 格式 GT 框，并把这些区域强制加入 keep mask。
5. 用 keep mask 处理图片，标签文件原样复制。
```

## FastSAM Usage / FastSAM 使用方式

| Scenario | Model | Prompt | Purpose | Evidence |
| --- | --- | --- | --- | --- |
| Midterm segmentation demo | `FastSAM-s.pt` | None, segment-all | Visualization comparison | `thesis_package/archive/root_scripts/build_midterm_seg_demos.py` |
| Road focus demo | `FastSAM-s.pt` | None vs `texts=['road']` | Compare automatic masks and road text filtering | `thesis_package/archive/root_scripts/build_road_focus_demos.py` |
| Current reproducible script | `FastSAM-s.pt` | `segment_all` or `text_road` | Reproducible visualization | `segmentation_pipeline/fastsam/scripts/run_fastsam_road_demo.py` |

| 场景 | 模型 | 提示词 | 用途 | 证据 |
| --- | --- | --- | --- | --- |
| 中期分割 demo | `FastSAM-s.pt` | 无提示，segment-all | 可视化对比 | `thesis_package/archive/root_scripts/build_midterm_seg_demos.py` |
| road focus demo | `FastSAM-s.pt` | 无提示 vs `texts=['road']` | 对比自动 mask 与 road 文本过滤 | `thesis_package/archive/root_scripts/build_road_focus_demos.py` |
| 当前可复现实验脚本 | `FastSAM-s.pt` | `segment_all` 或 `text_road` | 可复现实验可视化 | `segmentation_pipeline/fastsam/scripts/run_fastsam_road_demo.py` |

## Practical Difference / 实际区别

- PIDNet outputs semantic road-region labels from a road-scene parser.
- FastSAM outputs general segmentation masks and can optionally filter them with text prompts.
- The formal PIDRoad dataset uses PIDNet, not FastSAM.
- FastSAM remains a visualization baseline for comparing general segmentation behavior on road images.
- PIDNet road masking can remove damage pixels, so the PIDRoad generation script preserves annotated bbox regions.

- PIDNet 来自道路场景语义分割模型，输出 road 类语义区域。
- FastSAM 输出通用分割 mask，也可以用文本提示过滤 mask。
- 正式 PIDRoad 数据集使用 PIDNet，不使用 FastSAM。
- FastSAM 保留为道路图片上通用分割行为的可视化对照。
- PIDNet 路面 mask 可能切掉病害像素，因此 PIDRoad 生成脚本会保留人工标注 bbox 区域。
