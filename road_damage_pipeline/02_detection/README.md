# 02 Detection / 道路病害检测

This module contains detector training/evaluation wrappers and thesis metric evidence.

本模块保存检测模型训练、显式 val/test 评估入口和论文指标证据。

## Contents / 内容

- `scripts/train.py`: local training wrapper.
- `scripts/eval.py`: explicit val/test evaluation wrapper; writes `metrics_summary.json`.
- `scripts/infer_images.py`: packaged sample-image inference demo.
- `scripts/summarize_results.py`: flattens packaged metrics.
- `assets/final_eval_metrics/`: pulled final metric evidence.
- `samples/images/`: small selected RDD image subset.
- `weights/yolo11s_original_nondrone_noempty_best.pt`: small demo detector weight.

## Usage / 使用

Run the packaged non-drone, non-empty sample images:

运行打包的非无人机、非空标签样本图：

```bash
python road_damage_pipeline/02_detection/scripts/infer_images.py \
  --device cpu \
  --max-images 2
```

Run explicit evaluation:

运行显式 val/test 评估：

```bash
python road_damage_pipeline/02_detection/scripts/eval.py --help
```

Flatten packaged metrics:

整理已打包指标：

```bash
python road_damage_pipeline/02_detection/scripts/summarize_results.py
```

The report module calls this detector through the YOLO weight in `weights/`. Report generation uses model-predicted boxes only, not RDD txt ground-truth boxes.

报告模块使用 `weights/` 中的 YOLO 权重做预测。报告生成只使用模型预测框，不使用 RDD txt 人工标签框。

## YOLO source / YOLO 源码位置

The YOLO/RG11 source code is the repository itself, especially the root-level `ultralytics/` package. This module keeps configs, sample data, metrics, and wrappers; it does not duplicate the entire YOLO source inside `02_detection`.

YOLO/RG11 源码就是本仓库自身，主要是根目录的 `ultralytics/` 包。本模块只保存配置、样本、指标和封装入口，不在 `02_detection` 里面再复制一份完整 YOLO 源码。

If this repository is cloned normally, the scripts auto-detect the root `ultralytics/` package. If you run against a different fork, pass `--repo-root /path/to/fork`.

正常 clone 本仓库后，脚本会自动识别根目录 `ultralytics/`。如果要换成其他 fork，再手动传 `--repo-root /path/to/fork`。

## Sample rule / 小样本规则

`samples/images/` contains only selected RDD test images with non-empty labels and non-drone viewpoints. The CSV records the selected country, class, label path, lane-reference note, and bbox-overlap note.

`samples/images/` 只包含从 RDD test 中挑出的非空标签、非无人机视角图片。CSV 中记录国家、类别、标签路径、车道线/道路参考和 bbox 重叠说明。

## Metric rule / 指标规则

Use explicit `metrics_summary.json` for thesis tables. Do not use training `results.csv` as final per-class test evidence unless clearly marked as a training curve source.

论文最终表格使用显式评估得到的 `metrics_summary.json`。不要把训练过程 `results.csv` 当成最终逐类别 test 证据。
