# 01 Segmentation / 图像分割预处理

This module contains segmentation-preprocessing exploration evidence.

本模块保存“先分割道路再检测”的实验和可视化证据。

## Contents / 内容

- `assets/pidroad_strategy_visuals/`: original+bbox, PIDNet road mask, PIDNet road mask with bbox preserved, and triptych boards.
- `assets/pidnet_fastsam_comparison/`: PIDNet vs FastSAM all-mode vs FastSAM `road` text prompt comparison.
- `docs/PIDNET_FASTSAM_USAGE_NOTES.md`: verified usage notes.
- `scripts/`: scripts used to regenerate the visual evidence.
- `scripts/segment_uploaded_image.py`: live PIDNet road segmentation for a single uploaded image in the Web UI.

## Current conclusion / 当前结论

Formal PIDRoad data generation used PIDNet with Cityscapes road semantics and no prompt. FastSAM was used only for comparison/demo, not for formal dataset derivation. This module is not part of the final online report-generation path.

正式 PIDRoad 数据派生使用 PIDNet 的 Cityscapes road 语义分割，无提示词。FastSAM 只作为对比/演示，不作为正式数据派生方法。本模块不进入最终在线报告生成链路。

## Usage / 使用

Regenerate the packaged comparison assets when needed:

如需重新生成当前打包的对比图：

```bash
python road_damage_pipeline/01_segmentation/scripts/build_pidnet_fastsam_visual_comparison.py
python road_damage_pipeline/01_segmentation/scripts/build_pidroad_strategy_visuals.py
```

Run live segmentation for one uploaded image:

对单张上传图运行真实 PIDNet 道路分割：

```bash
python road_damage_pipeline/01_segmentation/scripts/segment_uploaded_image.py \
  --image road_damage_pipeline/02_detection/samples/images/raw/Japan_001608.jpg \
  --output-dir road_damage_pipeline/outputs/segmentation_live_demo \
  --device cpu
```

The Web UI uses this live path when the segmentation option is enabled for image uploads. It no longer copies packaged comparison boards as if they were the current upload.

Web UI 在图片上传时勾选分割探索，会走这条 live 路径，不再把打包对比图当成当前上传图的结果。

If external weights are missing, check the weight READMEs:

如果权重缺失，查看：

```text
weights/pidnet/README.md
weights/fastsam/README.md
```
