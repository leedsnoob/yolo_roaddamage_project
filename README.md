# YOLO Road Damage Project

This repository is the final runnable package for the road-damage graduation project. It keeps one YOLO source tree and one pipeline package.

## What Is Included

- `ultralytics/`: the customized YOLO11 source used by the detection pipeline.
- `road_damage_pipeline/`: five pipeline modules and the local Web workbench.
- `pyproject.toml`: editable install target for the local YOLO package.
- `RDD_RGNET_YOLO11_PORT.md` and `README_RDD_FINAL.md`: notes about the RG11/RDD-RGNet port and custom module compatibility.

## Pipeline Modules

| Module | Role |
|---|---|
| `01_segmentation` | PIDNet/FastSAM preprocessing exploration. This is optional visualization evidence, not part of the final report path. |
| `02_detection` | YOLO/RG11 image and frame detection. |
| `03_video_dedup` | Video detection tracking and cross-frame event deduplication. |
| `04_area_measurement` | Estimated area calculation with bbox rules, Depth Anything V2, and Metric3D. |
| `05_report_generation` | Qwen/SiliconFlow report generation from structured evidence. |
| `web` | FastAPI + React local workbench for image/video upload and step-by-step visualization. |

Final image flow:

```text
image -> detection -> area measurement -> report generation
```

Final video flow:

```text
video -> detection -> video dedup -> representative frames -> area measurement -> report generation
```

## Install

Use a Python environment with PyTorch, OpenCV, Ultralytics dependencies, FastAPI, and the depth-model dependencies needed by `04_area_measurement`.

```bash
python -m pip install -e .
python -m pip install -r road_damage_pipeline/requirements.txt
```

## Run Smoke Test

```bash
python road_damage_pipeline/scripts/smoke_test_pipeline.py --device cpu
```

## Run Web Workbench

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

Report generation uses SiliconFlow only when `SILICONFLOW_API_KEY` is set in the backend process environment. Do not commit API keys.

## Why The Repository Was Reduced

The original Ultralytics upstream documentation, Docker files, examples, tests, and CI workflows are not needed for this thesis package. They were removed from this repository to avoid unrelated GitHub Actions failures and to keep the project focused on the road-damage pipeline.

## License

This project keeps the upstream Ultralytics AGPL-3.0 license file because the local YOLO source is derived from Ultralytics.
