#!/usr/bin/env python3
"""Live M1/M3/M4 area estimation for model-predicted road-damage boxes.

This module is used by report generation. It intentionally does not use
FastSAM and does not use packaged per-class ratios. M3 and M4 run the actual
Depth Anything V2 / Metric3D depth models over the input image, then compute a
bbox-rectangle depth area corrected by the bbox empirical prior.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = MODULE_ROOT.parent
REPO_ROOT = PIPELINE_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent if REPO_ROOT.name == "ultralytics_yolo11_final" else PIPELINE_ROOT.parent

DEFAULT_DEPTH_REPO = WORKSPACE_ROOT / "area_experiments" / "area_measurement_v1" / "external" / "Depth-Anything-V2"
DEFAULT_DEPTH_CKPT = (
    WORKSPACE_ROOT
    / "area_experiments"
    / "area_measurement_v1"
    / "checkpoints"
    / "depth_anything_v2_metric_vkitti_vits.pth"
)

CLASS_ID_TO_CODE = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_ID_TO_EN = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}


@dataclass
class BBoxSpec:
    """A model-predicted bbox used as report evidence."""

    item_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass
class LiveAreaConfig:
    """Runtime configuration for live depth area estimation."""

    scale_factor: float = 0.01
    assumed_horizontal_fov_deg: float = 70.0
    depth_repo: Path = DEFAULT_DEPTH_REPO
    depth_checkpoint: Path = DEFAULT_DEPTH_CKPT
    depth_input_size: int = 518
    metric3d_model: str = "metric3d_vit_small"
    metric3d_input_height: int = 616
    metric3d_input_width: int = 1064
    device: str = "auto"


def resolve_device(device: str) -> str:
    """Resolve auto device without silently accepting broken MPS."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.ones(1, device="mps")
            return "mps"
        except Exception:
            return "cpu"
    return "cpu"


def class_code(class_id: int) -> str:
    return CLASS_ID_TO_CODE.get(int(class_id), f"class_{class_id}")


def class_name(class_id: int) -> str:
    return CLASS_ID_TO_EN.get(int(class_id), str(class_id))


def empirical_m1_area(class_id: int, width_px: float, height_px: float, scale_factor: float) -> tuple[float, float, float]:
    """Class-specific bbox rule used as the baseline area estimate."""
    width_m = max(width_px, 0.0) * scale_factor
    height_m = max(height_px, 0.0) * scale_factor
    if int(class_id) == 0:
        area_m2 = height_m * 0.8
    elif int(class_id) == 1:
        area_m2 = width_m * 1.2
    else:
        area_m2 = width_m * height_m / 3.0
    return width_m, height_m, area_m2


def empirical_prior(class_id: int, width_px: float, height_px: float, scale_factor: float) -> dict[str, float]:
    width_m, height_m, m1_area = empirical_m1_area(class_id, width_px, height_px, scale_factor)
    rectangle_area = max(width_m * height_m, 1e-12)
    return {
        "width_m": width_m,
        "height_m": height_m,
        "m1_area_m2": m1_area,
        "rectangle_area_m2": rectangle_area,
        "empirical_ratio": m1_area / rectangle_area,
    }


def bbox_mask(shape_hw: tuple[int, int], bbox_xyxy: list[float]) -> np.ndarray:
    height, width = shape_hw
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def normalize_depth_for_display(depth: np.ndarray) -> np.ndarray:
    """Convert metric depth to a stable false-color visualization."""
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((depth - low) / (high - low), 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def draw_bbox(image: np.ndarray, box: BBoxSpec, label: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box.bbox_xyxy]
    color = (0, 255, 255) if int(box.class_id) == 0 else (255, 255, 0) if int(box.class_id) == 1 else (0, 255, 0) if int(box.class_id) == 2 else (0, 128, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(image, (x1, max(0, y1 - 22)), (min(image.shape[1] - 1, x1 + 160), y1), color, -1)
    cv2.putText(image, label, (x1 + 3, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def depth_area(depth: np.ndarray, mask: np.ndarray, horizontal_fov_deg: float) -> tuple[float, float]:
    values = depth[(mask > 0) & np.isfinite(depth) & (depth > 0)]
    if values.size == 0:
        return 0.0, 0.0
    height, width = depth.shape[:2]
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    fy = fx
    pixel_areas = (values.astype(np.float64) ** 2) / (fx * fy)
    return float(pixel_areas.sum()), float(np.median(values))


class DepthAnythingRunner:
    """Depth Anything V2 metric VKITTI runner."""

    def __init__(self, config: LiveAreaConfig, device: str):
        self.config = config
        self.device = device
        self.model = None
        self._load()

    def _load(self) -> None:
        if not self.config.depth_repo.exists():
            raise RuntimeError(f"Depth Anything V2 repo not found: {self.config.depth_repo}")
        if not self.config.depth_checkpoint.exists():
            raise RuntimeError(f"Depth Anything V2 checkpoint not found: {self.config.depth_checkpoint}")

        metric_root = self.config.depth_repo / "metric_depth"
        if str(metric_root) not in sys.path:
            sys.path.insert(0, str(metric_root))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
            from torchvision.transforms import Compose
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to import Depth Anything V2 from {metric_root}: {type(exc).__name__}: {exc}") from exc

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        }
        model = DepthAnythingV2(**{**model_configs["vits"], "max_depth": 80})
        model.load_state_dict(torch.load(self.config.depth_checkpoint, map_location="cpu"))
        device = self.device

        def image2tensor_on_configured_device(self_model: Any, raw_image: np.ndarray, input_size: int = 518):
            transform = Compose(
                [
                    Resize(
                        width=input_size,
                        height=input_size,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )
            h, w = raw_image.shape[:2]
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            image = transform({"image": image})["image"]
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            return image, (h, w)

        # Upstream Depth Anything V2 hard-codes cuda/mps inside image2tensor().
        # Patch it so explicit --device cpu really keeps both input and weights on CPU.
        model.image2tensor = types.MethodType(image2tensor_on_configured_device, model)
        self.model = model.to(self.device).eval()

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            depth = self.model.infer_image(image_bgr, input_size=self.config.depth_input_size)
        if depth.shape[:2] != image_bgr.shape[:2]:
            depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)


class Metric3DRunner:
    """Metric3D runner with the non-CUDA decoder patch used in experiments."""

    def __init__(self, config: LiveAreaConfig, device: str):
        self.config = config
        self.device = device
        self.model = None
        self._load()

    def _load(self) -> None:
        try:
            model = torch.hub.load("yvanyin/metric3d", self.config.metric3d_model, pretrain=True, trust_repo=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load Metric3D via torch.hub: {type(exc).__name__}: {exc}") from exc
        self.model = model
        self._patch_cuda_hardcode_for_non_cuda()
        self.model = self.model.to(self.device).eval()

    def _patch_cuda_hardcode_for_non_cuda(self) -> None:
        if self.device.startswith("cuda") or self.model is None:
            return
        decoder = getattr(getattr(self.model, "depth_model", None), "decoder", None)
        if decoder is None:
            return

        def get_bins(self: Any, bins_num: int) -> torch.Tensor:
            device = next(self.parameters()).device
            bins = torch.linspace(math.log(self.min_val), math.log(self.max_val), bins_num, device=device)
            return torch.exp(bins)

        decoder.get_bins = types.MethodType(get_bins, decoder)

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb_origin = image_bgr[:, :, ::-1]
        original_h, original_w = rgb_origin.shape[:2]
        focal_px = original_w / (2.0 * math.tan(math.radians(self.config.assumed_horizontal_fov_deg) / 2.0))
        intrinsic = [focal_px, focal_px, original_w / 2.0, original_h / 2.0]

        input_h, input_w = self.config.metric3d_input_height, self.config.metric3d_input_width
        scale = min(input_h / original_h, input_w / original_w)
        rgb = cv2.resize(rgb_origin, (int(original_w * scale), int(original_h * scale)), interpolation=cv2.INTER_LINEAR)
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

        padding = [123.675, 116.28, 103.53]
        resized_h, resized_w = rgb.shape[:2]
        pad_h = input_h - resized_h
        pad_w = input_w - resized_w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(
            rgb,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        mean_t = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std_t = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb_t = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb_t = ((rgb_t - mean_t) / std_t)[None].to(self.device)

        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({"input": rgb_t})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0] : pred_depth.shape[0] - pad_info[1],
            pad_info[2] : pred_depth.shape[1] - pad_info[3],
        ]
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :],
            (original_h, original_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        canonical_to_real_scale = intrinsic[0] / 1000.0
        pred_depth = torch.clamp(pred_depth * canonical_to_real_scale, 0, 300)
        return pred_depth.detach().cpu().numpy().astype(np.float32)


class LiveAreaEngine:
    """Reusable live M1/M3/M4 engine for predicted boxes."""

    def __init__(self, config: LiveAreaConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.depth_anything = DepthAnythingRunner(config, self.device)
        self.metric3d = Metric3DRunner(config, self.device)

    def estimate_image(self, image_path: Path, boxes: list[BBoxSpec]) -> dict[str, list[dict[str, Any]]]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image for live area estimation: {image_path}")
        if not boxes:
            return {}
        depth_anything_depth = self.depth_anything.infer(image)
        metric3d_depth = self.metric3d.infer(image)
        return {
            box.item_id: self._estimate_box(box, image.shape[:2], depth_anything_depth, metric3d_depth)
            for box in boxes
        }

    def estimate_image_with_visuals(
        self,
        image_path: Path,
        boxes: list[BBoxSpec],
        visual_output_dir: Path,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
        """Estimate areas and write depth/area visual evidence for one image."""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image for live area estimation: {image_path}")
        visual_output_dir.mkdir(parents=True, exist_ok=True)
        if not boxes:
            return {}, {}

        depth_anything_depth = self.depth_anything.infer(image)
        metric3d_depth = self.metric3d.infer(image)
        estimates = {
            box.item_id: self._estimate_box(box, image.shape[:2], depth_anything_depth, metric3d_depth)
            for box in boxes
        }

        stem = image_path.stem
        depth_anything_vis = normalize_depth_for_display(depth_anything_depth)
        metric3d_vis = normalize_depth_for_display(metric3d_depth)
        bbox_vis = image.copy()
        for box in boxes:
            label = f"{class_code(box.class_id)} {box.confidence:.2f}"
            draw_bbox(bbox_vis, box, label)
            draw_bbox(depth_anything_vis, box, label)
            draw_bbox(metric3d_vis, box, label)

        depth_anything_path = visual_output_dir / f"{stem}_depth_anything.jpg"
        metric3d_path = visual_output_dir / f"{stem}_metric3d.jpg"
        board_path = visual_output_dir / f"{stem}_area_board.jpg"
        cv2.imwrite(str(depth_anything_path), depth_anything_vis)
        cv2.imwrite(str(metric3d_path), metric3d_vis)
        cv2.imwrite(str(board_path), self._make_area_board(bbox_vis, depth_anything_vis, metric3d_vis, boxes, estimates, stem))
        return estimates, {
            "depth_anything": str(depth_anything_path),
            "metric3d": str(metric3d_path),
            "area_board": str(board_path),
        }

    def _make_area_board(
        self,
        bbox_vis: np.ndarray,
        depth_anything_vis: np.ndarray,
        metric3d_vis: np.ndarray,
        boxes: list[BBoxSpec],
        estimates: dict[str, list[dict[str, Any]]],
        title: str,
    ) -> np.ndarray:
        panel_w, panel_h = 520, 390

        def resize_panel(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        bbox_panel = resize_panel(bbox_vis)
        da_panel = resize_panel(depth_anything_vis)
        metric_panel = resize_panel(metric3d_vis)
        text_panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
        cv2.rectangle(text_panel, (0, 0), (panel_w, 28), (255, 255, 255), -1)
        cv2.putText(text_panel, "Area estimates", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(text_panel, f"{title}: estimated area", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(text_panel, "M1=bbox empirical | M3=Depth Anything | M4=Metric3D", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 70, 70), 1, cv2.LINE_AA)
        y = 118
        for idx, box in enumerate(boxes[:12], start=1):
            areas = estimates.get(box.item_id, [])
            by_id = {row["method_id"]: row["estimated_area_m2"] for row in areas}
            line = (
                f"{idx:02d} {class_code(box.class_id)} conf={box.confidence:.2f} "
                f"M1={by_id.get('M1', 0):.3f} M3={by_id.get('M3', 0):.3f} M4={by_id.get('M4', 0):.3f} m2"
            )
            cv2.putText(text_panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
            y += 24
            if y > panel_h - 48:
                remaining = len(boxes) - idx
                if remaining > 0:
                    cv2.putText(text_panel, f"... {remaining} more detections in CSV/JSON", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
                break
        cv2.putText(text_panel, "All values are estimated, not physical ground truth.", (18, panel_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 180), 1, cv2.LINE_AA)

        for panel, label in ((bbox_panel, "Predicted bbox"), (da_panel, "Depth Anything V2"), (metric_panel, "Metric3D")):
            cv2.rectangle(panel, (0, 0), (panel_w, 28), (255, 255, 255), -1)
            cv2.putText(panel, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
        return np.vstack([np.hstack([bbox_panel, text_panel]), np.hstack([da_panel, metric_panel])])

    def _estimate_box(
        self,
        box: BBoxSpec,
        shape_hw: tuple[int, int],
        depth_anything_depth: np.ndarray,
        metric3d_depth: np.ndarray,
    ) -> list[dict[str, Any]]:
        x1, y1, x2, y2 = box.bbox_xyxy
        width_px = max(float(x2) - float(x1), 0.0)
        height_px = max(float(y2) - float(y1), 0.0)
        prior = empirical_prior(box.class_id, width_px, height_px, self.config.scale_factor)
        mask = bbox_mask(shape_hw, box.bbox_xyxy)
        mask_pixels = int(mask.sum())

        raw_m3, median_m3 = depth_area(depth_anything_depth, mask, self.config.assumed_horizontal_fov_deg)
        raw_m4, median_m4 = depth_area(metric3d_depth, mask, self.config.assumed_horizontal_fov_deg)
        m3 = raw_m3 * prior["empirical_ratio"]
        m4 = raw_m4 * prior["empirical_ratio"]

        common = {
            "width_px": round(width_px, 2),
            "height_px": round(height_px, 2),
            "width_m": round(prior["width_m"], 4),
            "height_m": round(prior["height_m"], 4),
            "mask_source": "bbox_rectangle",
            "mask_pixels": mask_pixels,
            "depth_area_is_assumption": True,
            "scale_assumption": "fixed 0.01 m/px; no camera calibration or lane-line calibration",
            "camera_assumption": f"horizontal_fov_deg={self.config.assumed_horizontal_fov_deg}",
        }
        return [
            {
                "method_id": "M1",
                "method_name": "bbox empirical rule",
                "estimated_area_m2": round(prior["m1_area_m2"], 6),
                "status": "success",
                "limitation": "bbox-based empirical estimate, not physical GT",
                **common,
            },
            {
                "method_id": "M3",
                "method_name": "Depth Anything V2 bbox-depth empirical area",
                "estimated_area_m2": round(m3, 6),
                "status": "success",
                "raw_depth_bbox_area_m2": round(raw_m3, 6),
                "depth_median_m": round(median_m3, 6),
                "empirical_prior_area_m2": round(prior["m1_area_m2"], 6),
                "empirical_ratio": round(prior["empirical_ratio"], 6),
                "limitation": "Depth Anything V2 metric depth with assumed FOV; estimated area only",
                **common,
            },
            {
                "method_id": "M4",
                "method_name": "Metric3D bbox-depth empirical area",
                "estimated_area_m2": round(m4, 6),
                "status": "success",
                "raw_depth_bbox_area_m2": round(raw_m4, 6),
                "depth_median_m": round(median_m4, 6),
                "empirical_prior_area_m2": round(prior["m1_area_m2"], 6),
                "empirical_ratio": round(prior["empirical_ratio"], 6),
                "limitation": "Metric3D depth with assumed FOV; estimated area only",
                **common,
            },
        ]


def build_config_from_args(args: Any) -> LiveAreaConfig:
    """Create config from argparse namespace used by report_pipeline.py."""
    depth_repo = getattr(args, "depth_repo", None) or DEFAULT_DEPTH_REPO
    depth_checkpoint = getattr(args, "depth_checkpoint", None) or DEFAULT_DEPTH_CKPT
    return LiveAreaConfig(
        scale_factor=float(getattr(args, "scale_factor", 0.01)),
        assumed_horizontal_fov_deg=float(getattr(args, "assumed_horizontal_fov_deg", 70.0)),
        depth_repo=Path(depth_repo),
        depth_checkpoint=Path(depth_checkpoint),
        depth_input_size=int(getattr(args, "depth_input_size", 518)),
        metric3d_model=str(getattr(args, "metric3d_model", "metric3d_vit_small")),
        metric3d_input_height=int(getattr(args, "metric3d_input_height", 616)),
        metric3d_input_width=int(getattr(args, "metric3d_input_width", 1064)),
        device=str(getattr(args, "device", "auto")),
    )
