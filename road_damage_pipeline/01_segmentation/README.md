# 01 Segmentation / 图像分割预处理

This module contains the segmentation-preprocessing evidence.

本模块保存“先分割道路再检测”的实验和可视化证据。

## Contents / 内容

- `assets/pidroad_strategy_visuals/`: original+bbox, PIDNet road mask, PIDNet road mask with bbox preserved, and triptych boards.
- `assets/pidnet_fastsam_comparison/`: PIDNet vs FastSAM all-mode vs FastSAM `road` text prompt comparison.
- `docs/PIDNET_FASTSAM_USAGE_NOTES.md`: verified usage notes.
- `scripts/`: scripts used to regenerate the visual evidence.

## Current conclusion / 当前结论

Formal PIDRoad data generation used PIDNet with Cityscapes road semantics and no prompt. FastSAM was used only for comparison/demo, not for formal dataset derivation.

正式 PIDRoad 数据派生使用 PIDNet 的 Cityscapes road 语义分割，无提示词。FastSAM 只作为对比/演示，不作为正式数据派生方法。

