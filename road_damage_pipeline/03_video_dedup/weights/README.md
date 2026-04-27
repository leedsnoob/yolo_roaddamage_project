# Weights / 权重说明

This folder keeps only small demo weights that are useful for local reproduction.

本目录只放适合本地复现的小型演示权重。

Included / 已包含：

- `yolo11s_original_nondrone_noempty_best.pt`
  - Source: `runs_final/deduped_runs/rdd2022_aggregate_noempty_experiments/original_nondrone_noempty_yolo11s/weights/best.pt`
  - Use: default video inference demo on `samples/videos/3_dense_130_190.mp4`

Not included / 未放入：

- Large `m` weights above 100 MB.
- Full training run folders.
- Full RDD2022 dataset.

For thesis final evaluation, metrics are stored under:

- `paper_assets/final_eval_metrics/`

Large weights should be kept locally or uploaded with Git LFS if required.

大模型权重建议本地保存，或确实需要时用 Git LFS 管理。
