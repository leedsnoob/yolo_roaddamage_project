# Road Damage Inspection Pipeline

This repository contains the runnable code package and final dissertation PDF for an end-to-end road-damage inspection project. The project connects object detection, video event deduplication, estimated damage-area calculation, and Qwen3-VL report generation in a local inspection workflow.

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

## Dissertation Results

The tables below reproduce the main numerical results reported in the dissertation. See [`finalReport.pdf`](finalReport.pdf) for the full protocol, citations, figures, ablation discussion, and limitations.

### Dataset Protocols

| Protocol | Train images | Validation images | Test images | Role |
|---|---:|---:|---:|---|
| Filtered original-image multi-country protocol | 16,259 | 3,399 | 3,414 | Main detector comparison |
| Filtered Japan original-image protocol | 6,684 | 1,378 | 1,359 | Country-specific fine-tuning and validation |

The main detector protocol uses the vehicle/street-view labelled subset so that the comparison focuses on visible road-damage localisation and classification in the targeted inspection setting.

### Main Detector Comparison

Params are in millions and FLOPs are in GFLOPs. Rows inside the `n` and `s` blocks follow the dissertation ordering by `mAP50`.

| Model family | Scale | Params | FLOPs | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | - | 41.4 | 134.5 | - | - | 0.598 | 0.291 |
| RT-DETRv2 | s | 20.0 | 60.0 | - | 0.630 | 0.645 | 0.357 |
| Dynamic+Strip+WIoU | n | 3.2 | 13.3 | 0.679 | 0.609 | **0.649** | 0.357 |
| RG11 | n | 3.1 | 13.2 | 0.664 | **0.616** | 0.647 | 0.354 |
| RG11+Strip+WIoU | n | 3.2 | 13.3 | 0.672 | 0.604 | 0.646 | **0.358** |
| RG11+DCNv2+BiFPN | n | 3.1 | 12.9 | **0.684** | 0.602 | 0.645 | 0.355 |
| YOLO11 | n | 2.6 | 6.3 | 0.661 | 0.602 | 0.636 | 0.350 |
| YOLOv8 | n | 3.0 | 8.1 | 0.652 | 0.600 | 0.630 | 0.338 |
| RG11+DCNv2+BiFPN | s | 11.4 | 47.2 | 0.690 | 0.621 | **0.667** | **0.366** |
| Dynamic+Strip+WIoU | s | 11.8 | 48.8 | **0.691** | 0.617 | 0.660 | **0.366** |
| RG11 | s | 11.7 | 48.4 | 0.690 | 0.621 | 0.659 | 0.364 |
| RG11+Strip+WIoU | s | 11.8 | 48.8 | 0.676 | **0.631** | 0.659 | 0.365 |
| YOLO11 | s | 9.4 | 21.3 | 0.660 | 0.624 | 0.658 | 0.361 |
| YOLOv8 | s | 11.1 | 28.4 | 0.660 | 0.617 | 0.644 | 0.350 |

### Standard and Open Deep-channel `m` Variants

The `m`-scale experiments test whether additional capacity improves the detector. The results do not show a general capacity-driven improvement over the best `s`-scale result.

| Branch | Setting | P3/P4/P5 channels | Params | FLOPs | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RG11 | Standard `m` | 256/512/512 | 24.2 | 167.4 | 0.682 | **0.626** | 0.662 | 0.364 |
| RG11 | Open `m` | 128/256/1024 | 31.6 | 73.4 | **0.695** | 0.619 | **0.664** | **0.367** |
| RG11+DCNv2+BiFPN | Standard `m` | 256/512/512 | 23.9 | 162.8 | 0.673 | 0.617 | 0.655 | 0.361 |
| RG11+DCNv2+BiFPN | Open `m` | 128/256/1024 | 30.6 | 71.5 | **0.691** | **0.621** | **0.662** | **0.365** |
| RG11+Strip+WIoU | Standard `m` | 256/512/512 | 24.5 | 168.3 | - | - | - | - |
| RG11+Strip+WIoU | Open `m` | 128/256/1024 | 32.1 | 73.9 | 0.680 | **0.625** | 0.658 | 0.364 |
| Dynamic+Strip+WIoU | Standard `m` | 256/512/512 | 24.5 | 168.3 | 0.671 | **0.633** | 0.657 | 0.362 |
| Dynamic+Strip+WIoU | Open `m` | 128/256/1024 | 32.1 | 73.9 | **0.683** | 0.624 | **0.660** | **0.364** |

### Japan Fine-tuning

The Japan-only protocol is a target-country check using the corresponding multi-country weights as the starting point.

| Model | Precision | Recall | mAP50 | mAP50-95 | Delta mAP50 |
|---|---:|---:|---:|---:|---:|
| Japan FT RG11 original `n` | 0.609 | **0.605** | 0.601 | 0.296 | 0.000 |
| Japan FT DCNv2+BiFPN `n` | 0.603 | 0.597 | 0.595 | 0.297 | -0.006 |
| Japan FT Strip+WIoU `n` | 0.607 | 0.599 | 0.599 | 0.296 | -0.002 |
| Japan FT Dynamic+Strip+WIoU `n` | **0.652** | 0.580 | **0.619** | **0.308** | **+0.018** |

### Video Deduplication Case Study

The video experiment uses a one-minute PathCare road-fault segment. It is an engineering comparison using proxy metrics because event-level video ground truth is not available.

| Tracker | FPS | Detections | Events | Fragmentation | Hits/event |
|---|---:|---:|---:|---:|---:|
| ByteTrack | **11.019** | 728 | 582 | 218 | 1.251 |
| BoT-SORT | 8.957 | 741 | 627 | 335 | 1.182 |
| DeepOC-SORT-lite | 7.602 | 809 | **430** | **33** | **1.881** |

### System-level Validation Examples

| Input type | Recorded evidence | Output checked |
|---|---|---|
| Single-image report | 1 image, 4 predicted D00 detections | Bbox visualisation, depth visualisations, area summary, and generated inspection report |
| Video report | 60.018 s video, 1,439 frames, 728 detections, 582 ByteTrack events | Track-event summary, representative frames, event-level area summaries, and generated video report |

Area figures in the report are estimates calculated from predicted boxes, fixed scale and field-of-view settings, and monocular depth cues. Measured physical damage areas are outside the available project evidence. Qwen3-VL is used downstream of structured evidence; it writes inspection-style reports from detection, event, and area records rather than performing detection itself.

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
- area outputs are estimates under the stated scale and field-of-view settings, with calibrated physical measurement outside the project evidence;
- Qwen3-VL report text is a human-review aid that needs checking against upstream structured evidence.

## License

This repository keeps the upstream Ultralytics AGPL-3.0 license file because the local YOLO source is derived from Ultralytics.
