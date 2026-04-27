# Road Damage Pipeline / 道路病害工程 Pipeline

This is the standalone thesis pipeline package. It separates the work into four independent modules instead of mixing detection, video deduplication, segmentation, and area estimation in one video folder.

这是毕设最终工程化 pipeline 目录。这里把四个模块拆开维护，不再把检测、视频去重、图像分割和面积估计混在同一个 video 文件夹里。

## Modules / 模块

| Module | Purpose | Key outputs |
|---|---|---|
| `01_segmentation` | PIDNet/FastSAM preprocessing exploration | PIDRoad strategy triptychs, PIDNet vs FastSAM comparison |
| `02_detection` | YOLO/RG11 detector training and explicit val/test evidence | final evaluation metrics, small RDD image samples, train/eval scripts |
| `03_video_dedup` | Video inference and cross-frame event deduplication | tracker comparison CSVs, dense-frame visual boards, keyframes |
| `04_area_measurement` | Area-estimation exploration on selected RDD GT boxes | four-method area tables, per-image boards, FastSAM crop diagnostics |

## Boundary / 边界

- `02_detection` is about frame/image-level detection metrics.
- `03_video_dedup` is about turning repeated video-frame detections into event-level counts.
- `04_area_measurement` is estimated area only. There is no camera calibration or physical GT area.
- `01_segmentation` records why PIDNet was used for PIDRoad preprocessing and why FastSAM was not used as the formal dataset-generation method.

- `02_detection` 只负责图像级/帧级检测指标。
- `03_video_dedup` 只负责把视频连续帧重复检测合并成事件级统计。
- `04_area_measurement` 只是估计面积，没有相机标定和真实物理面积 GT。
- `01_segmentation` 记录为什么正式 PIDRoad 数据派生采用 PIDNet，而不是 FastSAM。

## Legacy Source / 旧目录说明

`road_damage_video_pipeline/` is kept as a compatibility source while this package is being stabilized. New thesis-facing organization should use `road_damage_pipeline/`.

`road_damage_video_pipeline/` 暂时保留为兼容来源。之后论文展示、GitHub README 和前端页面都应优先引用 `road_damage_pipeline/`。

## Environment / 环境

Install the shared dependencies first:

先安装通用依赖：

```bash
python -m pip install -r road_damage_pipeline/requirements.txt
python -m pip install -e ultralytics_yolo11_final
```

If you use the existing local YOLO environment, run scripts with:

如果复用当前本机 YOLO 环境，可使用：

```bash
video_damage_analytics/.conda-yolo311/bin/python road_damage_pipeline/03_video_dedup/scripts/infer_video.py --help
```
