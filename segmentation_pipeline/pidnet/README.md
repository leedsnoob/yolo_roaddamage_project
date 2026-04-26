# PIDNet Road-Prior Pipeline

This subfolder keeps the formal PIDRoad dataset-generation pipeline.

本子目录保存正式 PIDRoad 数据派生流程。

## Contents / 内容

- `source/`: Minimal PIDNet source copy with `models/`, `configs/`, `datasets/`, `tools/`, and `utils/`.
- `scripts/prepare_pidnet_road_dataset.py`: Reproducible PIDRoad generation script.
- `weights/PIDNet_L_Cityscapes_test.pt`: PIDNet-L Cityscapes test checkpoint.

- `source/`：PIDNet 最小源码副本，包含 `models/`、`configs/`、`datasets/`、`tools/`、`utils/`。
- `scripts/prepare_pidnet_road_dataset.py`：当前可复现的 PIDRoad 生成脚本。
- `weights/PIDNet_L_Cityscapes_test.pt`：PIDNet-L Cityscapes test 权重。

## Default Model / 默认模型

```text
architecture: pidnet-l
num_classes: 19
pretraining / finetuning: Cityscapes
checkpoint: weights/PIDNet_L_Cityscapes_test.pt
road_class_id: 0
```

## Command / 命令

```bash
python segmentation_pipeline/pidnet/scripts/prepare_pidnet_road_dataset.py \
  --input-root RDD2022/RDD_SPLIT \
  --output-root segmentation_pipeline/outputs/RDD_SPLIT_pidroad_nondrone \
  --yaml-out segmentation_pipeline/outputs/rdd2022_pidroad_nondrone.yaml \
  --splits train val \
  --include-test \
  --exclude-drone
```

## Limitation / 限制

This generation script uses RDD GT boxes to preserve damage regions. It is a controlled preprocessing experiment, not a pure deployment-time preprocessing pipeline.

该生成脚本使用 RDD GT 框保留病害区域。它是受控预处理实验，不是纯部署场景下的前处理流程。
