# Road Damage Pipeline / 道路病害工程 Pipeline

This is the standalone thesis pipeline package. It separates the work into five independent modules. The final runnable demo path is `02_detection -> 03_video_dedup -> 04_area_measurement -> 05_report_generation`.

The YOLO source code is not duplicated inside `02_detection`; it lives at the repository root in `ultralytics/`. The packaged scripts auto-detect this root, so a normal clone can run the demo without cloning another YOLO repo.

这是毕设最终工程化 pipeline 目录。这里把五个模块拆开维护。最终可运行 demo 链路是 `02_detection -> 03_video_dedup -> 04_area_measurement -> 05_report_generation`。

YOLO 源码没有重复放进 `02_detection`，而是在仓库根目录的 `ultralytics/`。打包脚本会自动识别这个根目录，所以正常 clone 本仓库后不需要再额外 clone 一个 YOLO 仓库即可运行 demo。

## Modules / 模块

| Module | Purpose | Key outputs |
|---|---|---|
| `01_segmentation` | PIDNet/FastSAM preprocessing exploration | PIDRoad strategy triptychs, PIDNet vs FastSAM comparison |
| `02_detection` | YOLO/RG11 detector training and explicit val/test evidence | final evaluation metrics, small RDD image samples, train/eval scripts |
| `03_video_dedup` | Video inference and cross-frame event deduplication | tracker comparison CSVs, dense-frame visual boards, keyframes |
| `04_area_measurement` | Area-estimation exploration on selected RDD GT boxes | four-method area tables, per-image boards, FastSAM crop diagnostics |
| `05_report_generation` | Evidence-grounded intelligent report generation | report-agent design, input schema, writer/verifier prompts |

## Boundary / 边界

- `02_detection` is about frame/image-level detection metrics.
- `03_video_dedup` is about turning repeated video-frame detections into event-level counts.
- `04_area_measurement` is estimated area only. There is no camera calibration or physical GT area.
- `01_segmentation` records why PIDNet was used for PIDRoad preprocessing and why FastSAM was not used as the formal dataset-generation method.
- `05_report_generation` turns structured evidence into reports. It must not invent detections, counts, areas, or accuracy.
- The full paper-facing README with model diagrams, experiment tables, and frontend instructions will be completed after the thesis text stabilizes.

- `02_detection` 只负责图像级/帧级检测指标。
- `03_video_dedup` 只负责把视频连续帧重复检测合并成事件级统计。
- `04_area_measurement` 只是估计面积，没有相机标定和真实物理面积 GT。
- `01_segmentation` 记录为什么正式 PIDRoad 数据派生采用 PIDNet，而不是 FastSAM。
- `05_report_generation` 只把结构化证据转成报告，不能编造检测、数量、面积或准确率。
- 包含网络结构图、实验数据总表和前端启动方式的最终论文级 README，等论文内容稳定后再补。

## Legacy Source / 旧目录说明

`road_damage_video_pipeline/` is kept as a compatibility source while this package is being stabilized. New thesis-facing organization should use `road_damage_pipeline/`.

`road_damage_video_pipeline/` 暂时保留为兼容来源。之后论文展示、GitHub README 和前端页面都应优先引用 `road_damage_pipeline/`。

## Environment / 环境

Install the shared dependencies first:

先安装通用依赖：

```bash
python -m pip install -r road_damage_pipeline/requirements.txt
python -m pip install -e .
```

If you use the existing local YOLO environment, run scripts with:

如果复用当前本机 YOLO 环境，可使用：

```bash
video_damage_analytics/.conda-yolo311/bin/python road_damage_pipeline/03_video_dedup/scripts/infer_video.py --help
```

## Smoke Test / 快速连通性测试

Run this after cloning the repository and installing dependencies:

clone 并安装依赖后，可先运行：

```bash
python road_damage_pipeline/scripts/smoke_test_pipeline.py --device cpu
```

This checks:

这会检查：

- the small image sample set is non-drone and non-empty;
- `02_detection` can load the packaged YOLO source and demo weight;
- `03_video_dedup` can generate short video detections and track events;
- `04_area_measurement` packaged M1/M3/M4 assets are present;
- `05_report_generation` can build image/video report inputs without calling the API.

## Report Demo / 报告生成 Demo

The report module uses SiliconFlow through `SILICONFLOW_API_KEY`. Do not write API keys into this repository.

报告模块通过 `SILICONFLOW_API_KEY` 调用 SiliconFlow。不要把 API key 写入仓库。

```bash
export SILICONFLOW_API_KEY="your_api_key_here"

python road_damage_pipeline/05_report_generation/scripts/report_pipeline.py \
  --mode image \
  --call-api

python road_damage_pipeline/05_report_generation/scripts/report_pipeline.py \
  --mode video \
  --call-api
```
