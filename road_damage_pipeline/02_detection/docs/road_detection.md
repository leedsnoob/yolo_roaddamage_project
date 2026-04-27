# Road Damage Detection / 道路病害检测

## Purpose / 目的

This part contains the detector training, explicit validation/test evaluation, and paper-ready metric summaries.

本部分包含检测模型训练、显式 val/test 评估和论文可用指标汇总。

## Dataset Protocol / 数据协议

Main detection experiments use:

主检测实验使用：

- Dataset: RDD2022
- Protocol: original non-drone no-empty
- Classes: `D00`, `D10`, `D20`, `D40`
- Class mapping:
  - `0 -> D00 -> Longitudinal Crack`
  - `1 -> D10 -> Transverse Crack`
  - `2 -> D20 -> Alligator Crack`
  - `3 -> D40 -> Pothole`

Japan fine-tuning experiments use:

Japan 微调实验使用：

- Protocol: original Japan no-empty
- Evaluation split: explicit `val` and `test`

## Scripts / 脚本

- Train: `scripts/train.py`
- Eval: `scripts/eval.py`
- Summary: `scripts/summarize_results.py`

Example:

```bash
python road_damage_pipeline/02_detection/scripts/eval.py \
  --config road_damage_pipeline/02_detection/configs/eval_default.yaml
```

## Metric Source Rule / 指标来源规则

For thesis tables, use explicit evaluation outputs:

论文表格优先使用显式评估输出：

- `metrics_summary.json`
- `final_eval_requirements.csv`
- `final_eval_metrics_flat.csv`

Do not use training `results.csv` as the final per-class table unless it is clearly marked as a training/validation curve source.

不要把训练过程中的 `results.csv` 当作最终逐类别表格，除非明确说明它只是训练曲线来源。

## Current Coverage / 当前覆盖

The packaged evidence contains:

当前证据包包含：

- `58` explicit `metrics_summary.json` files
- `290 / 300` required metric rows available
- Missing rows: `YOLO11m` val/test only

## Evidence / 证据文件

- Final evaluation metrics: `assets/final_eval_metrics/`
- Demo image samples: `samples/images/`
- Detector code fork: repository root `ultralytics/`
- Custom RGNet notes: `RDD_RGNET_YOLO11_PORT.md`
