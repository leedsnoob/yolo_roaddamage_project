# Segmentation Pipeline: PIDNet vs FastSAM

This folder records the road-segmentation preprocessing exploration used in the project. It keeps PIDNet and FastSAM in one reproducible place for later visualization, demo pages, and experiment tracing.

本目录用于整理项目中“先分割路面再检测病害”的预处理探索流程。PIDNet 和 FastSAM 被放在同一个可复现实验目录中，方便后续前端展示、可视化页面和实验追溯。

## Scope

- The formal PIDRoad-derived dataset was generated with `PIDNet-L + Cityscapes`.
- PIDNet is not prompt-based. It predicts Cityscapes semantic classes and uses the `road` class, class id `0`.
- The PIDRoad generation script preserves YOLO GT bounding-box regions with padding to avoid cutting damage regions.
- FastSAM was used only for visualization and comparison, not for formal PIDRoad dataset generation.
- FastSAM comparison supports two modes: `segment-all` without prompts and `text prompt = road`.

## 范围

- 正式 PIDRoad 派生数据集使用 `PIDNet-L + Cityscapes` 生成。
- PIDNet 不是提示词模型。它输出 Cityscapes 语义分割类别，并取 `road` 类，类别 id 为 `0`。
- PIDRoad 生成脚本会额外保留 YOLO 人工标注 bbox 及 padding，避免病害区域被路面 mask 切掉。
- FastSAM 只用于可视化和对比，不参与正式 PIDRoad 数据集生成。
- FastSAM 对比支持两种模式：无提示 `segment-all` 和文本提示 `road`。

## Directory Layout / 目录结构

```text
segmentation_pipeline/
  README.md
  configs/default_segmentation_pipeline.yaml
  docs/PIDNET_FASTSAM_USAGE_NOTES.md
  pidnet/
    source/                         # Minimal PIDNet source copy / PIDNet 最小源码副本
    scripts/prepare_pidnet_road_dataset.py
    weights/PIDNet_L_Cityscapes_test.pt
  fastsam/
    source_ultralytics_fastsam/      # Ultralytics FastSAM adapter used by this project
    scripts/run_fastsam_road_demo.py
    weights/FastSAM-s.pt
  outputs/
```

## Weights / 权重

Large checkpoints are intentionally not tracked by Git. Keep them locally under the paths below before running the scripts.

大权重文件不会进入 Git 仓库。运行脚本前，请将权重放到下表对应的本地路径。

| Model / 模型 | Local path / 本地路径 | Status / 状态 | SHA256 |
| --- | --- | --- | --- |
| PIDNet-L Cityscapes test | `pidnet/weights/PIDNet_L_Cityscapes_test.pt` | Local only, not committed / 仅本地保存，不提交 | `7d90237c8b1c633283c77acd60dc4deeed5afacb872cb5c212493cae1e42b1a7` |
| FastSAM-s | `fastsam/weights/FastSAM-s.pt` | Local only, not committed / 仅本地保存，不提交 | `c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844` |

Download sources if the local files are missing:

如果本地文件缺失，可从以下来源重新下载：

- PIDNet official repo / 官方仓库：<https://github.com/XuJiacong/PIDNet>
- Expected PIDNet checkpoint / PIDNet 权重文件名：`PIDNet_L_Cityscapes_test.pt`
- FastSAM official repo / 官方仓库：<https://github.com/CASIA-IVA-Lab/FastSAM>
- Ultralytics FastSAM implementation / Ultralytics FastSAM 实现入口：`ultralytics.models.fastsam`

## PIDNet Dataset Generation / PIDNet 数据派生

Smoke test:

快速测试：

```bash
python segmentation_pipeline/pidnet/scripts/prepare_pidnet_road_dataset.py \
  --input-root RDD2022/RDD_SPLIT \
  --output-root segmentation_pipeline/outputs/RDD_SPLIT_pidroad_smoketest \
  --yaml-out segmentation_pipeline/outputs/rdd2022_pidroad_smoketest.yaml \
  --splits val \
  --limit 8 \
  --preview-count 4
```

Full generation can add `--splits train val --include-test`.

完整生成可以增加 `--splits train val --include-test`。

Important limitation: this dataset-generation script uses existing GT boxes to preserve damage regions. It is useful for controlled preprocessing experiments, but it is not a pure deployment-time preprocessing pipeline.

重要限制：该数据生成脚本使用已有 GT 框来保留病害区域。因此它适合做受控预处理实验，但不是纯部署场景下可直接使用的前处理流程。

## PIDRoad Strategy Visualization / PIDRoad 策略可视化

Build per-sample triptychs for the three preprocessing stages:

生成每个样本一张三联图，对应三个预处理阶段：

1. Original image with GT bbox / 原图 + GT 框
2. PIDNet road-only image / PIDNet 仅保留 road 区域
3. PIDNet road plus GT bbox-preserved image / PIDNet road + 保留 GT bbox 区域

```bash
python segmentation_pipeline/scripts/build_pidroad_strategy_visuals.py --device auto
```

Outputs:

输出目录：

```text
segmentation_pipeline/outputs/pidroad_strategy_visuals/
  01_original_bbox/
  02_pidnet_road/
  03_pidnet_road_bbox_preserve/
  04_triptych/
  strategy_metrics.csv
  summary.json
```

## FastSAM Visualization / FastSAM 可视化

Single-image comparison:

单图对比：

```bash
python segmentation_pipeline/fastsam/scripts/run_fastsam_road_demo.py \
  --image RDD2022/RDD_SPLIT/test/images/Japan_001608.jpg \
  --mode both \
  --output-dir segmentation_pipeline/outputs/fastsam_road_demo
```

`segment-all` runs automatic segmentation for all candidate regions. `text_road` applies the text prompt `road` to filter masks. Both are comparison modes only.

`segment-all` 会自动分割所有候选区域。`text_road` 会使用文本提示 `road` 过滤 mask。两者都只作为对比模式。

Note: `text_road` uses the CLIP branch in Ultralytics FastSAM. The first run may install the CLIP dependency and download CLIP weights.

注意：`text_road` 会调用 Ultralytics FastSAM 的 CLIP 分支。首次运行可能会安装 CLIP 依赖并下载 CLIP 权重。
