# Road Damage Inspection Pipeline

This repository contains the runnable code package and final dissertation PDF for an end-to-end road-damage inspection project. The project connects object detection, video event deduplication, assumption-based area estimation, and Qwen3-VL report generation in a local inspection workflow.

## Dissertation

- Final report PDF: [`finalReport.pdf`](finalReport.pdf)
- Main dataset: RDD2022 multi-country road-damage images
- Video case study: one-minute road-damage segment used for event deduplication and report-generation validation

The dissertation explains the experimental protocol, model comparisons, system design, limitations, and future work. This README is only a repository guide.

## Repository Contents

| Path | Purpose |
|---|---|
| `ultralytics/` | Local YOLO source tree used by the detection pipeline. |
| `road_damage_pipeline/` | Canonical package for segmentation exploration, detection, video deduplication, area estimation, report generation, and the web workbench. |
| `pyproject.toml` | Editable install target for the local package. |
| `RDD_RGNET_YOLO11_PORT.md` | Notes on the RG11/RDD-RGNet port to YOLO11. |
| `README_RDD_FINAL.md` | Additional notes on the final RDD setup. |
| `finalReport.pdf` | Final dissertation PDF. |

## Pipeline Overview

The final image workflow is:

```text
image -> detection -> area estimation -> report generation
```

The final video workflow is:

```text
video -> frame detection -> tracking/deduplication -> representative frames -> area estimation -> report generation
```

The modules are kept separate so that detector outputs, event summaries, area estimates, and generated reports can be audited independently.

## Modules

| Module | Role |
|---|---|
| `01_segmentation` | PIDNet/FastSAM road-region preprocessing exploration. This is analysis evidence, not the final runtime input path. |
| `02_detection` | YOLOv8, YOLO11, RG11, and RG11-based detector training/evaluation scripts. |
| `03_video_dedup` | Video detection tracking and repeated-frame deduplication. |
| `04_area_measurement` | Rule-based bbox geometry, Depth Anything V2-assisted, and Metric3D-assisted area estimates. |
| `05_report_generation` | Evidence-grounded Qwen3-VL inspection report generation through SiliconFlow. |
| `web` | FastAPI backend and React/Vite frontend for local image/video inspection. |

## Main Findings Reported in the Dissertation

- PIDRoad-style preprocessing improves the matched processed test view, but the final runtime path starts from object detection because damage boxes are not available before inference.
- Detector improvements are scale- and class-dependent. Dynamic+Strip+WIoU gives the strongest `n`-scale `mAP50` in the main comparison (`0.649`), while RG11+DCNv2+BiFPN is strongest at `s` scale (`mAP50=0.667`, `mAP50-95=0.366`).
- `m`-scale experiments do not show that larger capacity alone improves road-damage detection. Standard RG11 `m` reaches `mAP50=0.662`, below the best `s`-scale result, while open deep-channel `m` variants give only small route-relative gains.
- Japan fine-tuning provides a separate target-country check. Dynamic+Strip+WIoU improves the Japan fine-tuning `mAP50` from `0.601` to `0.619`, while recall decreases from `0.605` to `0.580`.
- Video deduplication is evaluated as an engineering case study without event-level ground truth. ByteTrack is fastest in the recorded case (`11.019 FPS`), while DeepOC-SORT-lite produces fewer unique events and lower fragmentation at a speed cost.
- Area values are assumption-based estimates derived from predicted boxes, fixed scale/FOV settings, and monocular depth cues. Physical interpretation would require camera or lane calibration.
- Qwen3-VL is used downstream of structured evidence. It writes inspection-style reports from detection, event, and area records rather than performing detection itself.

## Install

Use a Python environment with PyTorch, OpenCV, Ultralytics dependencies, FastAPI, and the depth-model dependencies needed by `04_area_measurement`.

```bash
python -m pip install -e .
python -m pip install -r road_damage_pipeline/requirements.txt
```

## Smoke Test

```bash
python road_damage_pipeline/scripts/smoke_test_pipeline.py --device cpu
```

This checks that the packaged YOLO source, demo pipeline modules, and report-input builders can be imported and connected.

## Web Workbench

Backend:

```bash
ROAD_DAMAGE_PIPELINE_PYTHON=/path/to/python \
  python -m uvicorn road_damage_pipeline.web.backend.app.main:app \
  --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd road_damage_pipeline/web/frontend
npm install
npm run dev
```

Report generation calls SiliconFlow only when `SILICONFLOW_API_KEY` is set in the backend process environment. Do not commit API keys.

## Evidence Boundary

The repository is a thesis code package, not a deployed road-maintenance decision system. Runtime artifacts such as detections, event summaries, area tables, visualizations, and generated reports are written locally and are not committed by default. External datasets, pretrained weights, third-party models, and large experiment outputs are documented in the dissertation rather than bundled here.

The most important interpretation limits are:

- detector results depend on the stated dataset split, scale, and protocol;
- video deduplication results use proxy metrics because event-level video ground truth is not available;
- area outputs are assumption-based estimates and require calibration before physical interpretation;
- Qwen3-VL report text remains a human-review aid and should be checked against upstream structured evidence.

## License

This repository keeps the upstream Ultralytics AGPL-3.0 license file because the local YOLO source is derived from Ultralytics.
