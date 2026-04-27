# 02 Detection / 道路病害检测

This module contains detector training/evaluation wrappers and thesis metric evidence.

本模块保存检测模型训练、显式 val/test 评估入口和论文指标证据。

## Contents / 内容

- `scripts/train.py`: local training wrapper.
- `scripts/eval.py`: explicit val/test evaluation wrapper; writes `metrics_summary.json`.
- `scripts/summarize_results.py`: flattens packaged metrics.
- `assets/final_eval_metrics/`: pulled final metric evidence.
- `samples/images/`: small selected RDD image subset.
- `weights/yolo11s_original_nondrone_noempty_best.pt`: small demo detector weight.

## Usage / 使用

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

## Metric rule / 指标规则

Use explicit `metrics_summary.json` for thesis tables. Do not use training `results.csv` as final per-class test evidence unless clearly marked as a training curve source.

论文最终表格使用显式评估得到的 `metrics_summary.json`。不要把训练过程 `results.csv` 当成最终逐类别 test 证据。
