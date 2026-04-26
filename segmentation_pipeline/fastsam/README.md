# FastSAM Comparison Pipeline

This subfolder keeps the FastSAM visualization pipeline. It is not used for formal PIDRoad dataset generation.

本子目录保存 FastSAM 可视化对比流程。它不用于正式 PIDRoad 数据集生成。

## Contents / 内容

- `source_ultralytics_fastsam/`: FastSAM adapter code from the local Ultralytics fork.
- `scripts/run_fastsam_road_demo.py`: Single-image FastSAM visualization script.
- `weights/FastSAM-s.pt`: FastSAM-s checkpoint.

- `source_ultralytics_fastsam/`：来自本地 Ultralytics fork 的 FastSAM 适配代码。
- `scripts/run_fastsam_road_demo.py`：单图 FastSAM 可视化脚本。
- `weights/FastSAM-s.pt`：FastSAM-s 权重。

## Modes / 模式

```text
segment_all:
  No prompt. Segment all candidate regions in the image.
  无提示词，对整张图做通用实例分割。

text_road:
  Use texts=['road'] for text-guided mask filtering.
  使用 texts=['road'] 做文本提示过滤。
```

`text_road` uses the Ultralytics CLIP branch. The first run may install the CLIP dependency and download CLIP weights.

`text_road` 会额外使用 Ultralytics 的 CLIP 分支。首次运行可能会安装 CLIP 依赖并下载 CLIP 权重。

## Example / 示例

```bash
python segmentation_pipeline/fastsam/scripts/run_fastsam_road_demo.py \
  --image RDD2022/RDD_SPLIT/test/images/Japan_001608.jpg \
  --mode both \
  --imgsz 768 \
  --conf 0.1 \
  --iou 0.9
```

Default output:

默认输出目录：

```text
segmentation_pipeline/outputs/fastsam_road_demo/
```
