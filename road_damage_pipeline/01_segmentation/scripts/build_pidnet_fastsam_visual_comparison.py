import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parent
PIDNET_ROOT = PIPELINE_ROOT / "pidnet" / "source"
PIDNET_WEIGHTS = PIPELINE_ROOT / "pidnet" / "weights" / "PIDNet_L_Cityscapes_test.pt"
FASTSAM_WEIGHTS = PIPELINE_ROOT / "fastsam" / "weights" / "FastSAM-s.pt"
ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics_yolo11_final"
IMAGE_ROOT = PROJECT_ROOT / "RDD2022" / "RDD_SPLIT" / "test" / "images"
OUTPUT_ROOT = PIPELINE_ROOT / "outputs" / "pidnet_fastsam_comparison"

SELECTED = [
    ("D002", "Czech_000201.jpg"),
    ("D005", "Czech_000194.jpg"),
    ("D008", "India_000090.jpg"),
    ("D011", "India_000118.jpg"),
    ("D014", "Japan_000035.jpg"),
    ("D017", "Japan_000076.jpg"),
    ("D020", "Norway_000031.jpg"),
    ("D023", "Norway_000059.jpg"),
    ("D026", "United_States_000046.jpg"),
    ("D029", "United_States_000083.jpg"),
    ("D032", "China_MotorBike_000036.jpg"),
    ("D035", "China_MotorBike_000160.jpg"),
]

PIDNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIDNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PIDNET_COLOR_MAP = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]
INSTANCE_COLORS = [
    (0, 255, 255),
    (255, 200, 0),
    (0, 220, 120),
    (255, 120, 120),
    (120, 180, 255),
    (180, 120, 255),
    (255, 220, 120),
    (120, 255, 220),
    (80, 160, 255),
    (220, 120, 80),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build PIDNet vs FastSAM visual comparison boards and metrics.")
    parser.add_argument("--image-root", default=str(IMAGE_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--pidnet-root", default=str(PIDNET_ROOT))
    parser.add_argument("--pidnet-weights", default=str(PIDNET_WEIGHTS))
    parser.add_argument("--fastsam-weights", default=str(FASTSAM_WEIGHTS))
    parser.add_argument("--ultralytics-root", default=str(ULTRALYTICS_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fastsam-imgsz", type=int, default=768)
    parser.add_argument("--fastsam-conf", type=float, default=0.1)
    parser.add_argument("--fastsam-iou", type=float, default=0.9)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def resolve_torch_device(device: str):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_ultralytics_device(device: torch.device):
    if device.type == "cuda":
        return 0
    return device.type


def setup_imports(pidnet_root: Path, ultralytics_root: Path):
    for path in (pidnet_root, ultralytics_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_pidnet(pidnet_root: Path, weights: Path, device: torch.device):
    setup_imports(pidnet_root, ULTRALYTICS_ROOT)
    import models

    model = models.pidnet.get_pred_model("pidnet-l", 19)
    pretrained = torch.load(weights, map_location="cpu")
    pretrained = pretrained["state_dict"] if "state_dict" in pretrained else pretrained
    model_dict = model.state_dict()
    pretrained = {
        key[6:]: value
        for key, value in pretrained.items()
        if key.startswith("model.") and key[6:] in model_dict and model_dict[key[6:]].shape == value.shape
    }
    model_dict.update(pretrained)
    model.load_state_dict(model_dict, strict=False)
    model.to(device).eval()
    return model


def load_fastsam(weights: Path, ultralytics_root: Path):
    if str(ultralytics_root) not in sys.path:
        sys.path.insert(0, str(ultralytics_root))
    from ultralytics import FastSAM

    return FastSAM(str(weights))


def pidnet_input_transform(image_bgr: np.ndarray):
    image = image_bgr.astype(np.float32)[:, :, ::-1] / 255.0
    image -= PIDNET_MEAN
    image /= PIDNET_STD
    return image.transpose((2, 0, 1)).copy()


def sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def pidnet_predict(model, image_bgr: np.ndarray, device: torch.device):
    tensor = torch.from_numpy(pidnet_input_transform(image_bgr)).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    aligned_height = int(np.ceil(height / 32.0) * 32)
    aligned_width = int(np.ceil(width / 32.0) * 32)
    if (aligned_height, aligned_width) != (height, width):
        tensor = F.interpolate(tensor, size=(aligned_height, aligned_width), mode="bilinear", align_corners=True)
    sync_if_needed(device)
    start = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    semantic = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(PIDNET_COLOR_MAP):
        semantic[pred == class_id] = color
    road_mask = (pred == 0).astype(np.uint8)
    return pred, road_mask, semantic, elapsed


def resize_mask(mask: np.ndarray, width: int, height: int):
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def render_instance_masks(masks, width: int, height: int):
    union = np.zeros((height, width), dtype=np.uint8)
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    if masks is None or len(masks) == 0:
        return union, vis, 0
    masks_np = masks.detach().float().cpu().numpy()
    areas = masks_np.sum(axis=(1, 2))
    order = np.argsort(areas)[::-1]
    for color_index, mask_index in enumerate(order.tolist()):
        mask = masks_np[mask_index] > 0
        if mask.shape != (height, width):
            mask = resize_mask(mask.astype(np.uint8), width, height) > 0
        union[mask] = 1
        vis[mask] = INSTANCE_COLORS[color_index % len(INSTANCE_COLORS)]
    return union, vis, int(len(order))


def fastsam_predict(model, image_path: Path, image_bgr: np.ndarray, args, device_name, texts=None):
    height, width = image_bgr.shape[:2]
    kwargs = dict(
        source=str(image_path),
        device=device_name,
        retina_masks=False,
        imgsz=args.fastsam_imgsz,
        conf=args.fastsam_conf,
        iou=args.fastsam_iou,
        verbose=False,
    )
    start = time.perf_counter()
    try:
        if texts is None:
            results = model.predict(**kwargs)
        else:
            results = model.predict(**kwargs, texts=texts)
    except RuntimeError:
        kwargs["device"] = "cpu"
        if texts is None:
            results = model.predict(**kwargs)
        else:
            results = model.predict(**kwargs, texts=texts)
    elapsed = time.perf_counter() - start
    masks = None if results[0].masks is None else results[0].masks.data
    union, vis, count = render_instance_masks(masks, width, height)
    return union, vis, count, elapsed


def mask_stats(mask: np.ndarray):
    mask = (mask > 0).astype(np.uint8)
    total = int(mask.size)
    pixels = int(mask.sum())
    coverage = pixels / total if total else 0.0
    if pixels == 0:
        return {
            "mask_pixels": 0,
            "coverage_ratio": 0.0,
            "component_count": 0,
            "largest_component_ratio": 0.0,
        }
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    largest = int(component_areas.max()) if component_areas.size else pixels
    return {
        "mask_pixels": pixels,
        "coverage_ratio": coverage,
        "component_count": int(max(0, num_labels - 1)),
        "largest_component_ratio": largest / pixels if pixels else 0.0,
    }


def overlay_mask(image_bgr: np.ndarray, mask_or_vis: np.ndarray, alpha: float = 0.42, color=(0, 255, 0)):
    if mask_or_vis.ndim == 2:
        vis = np.zeros_like(image_bgr)
        vis[mask_or_vis > 0] = color
    else:
        vis = mask_or_vis
    return cv2.addWeighted(image_bgr, 1.0 - alpha, vis, alpha, 0)


def add_title(image: np.ndarray, title: str):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (255, 255, 255), -1)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def build_triptych(original: np.ndarray, segmentation: np.ndarray, overlay: np.ndarray, caption: str):
    board = np.concatenate(
        [add_title(original, "Original"), add_title(segmentation, "Segmentation"), add_title(overlay, "Overlay")],
        axis=1,
    )
    footer = np.full((56, board.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(footer, caption, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    return np.concatenate([board, footer], axis=0)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    image_root = Path(args.image_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    pidnet_root = Path(args.pidnet_root).expanduser().resolve()
    ultralytics_root = Path(args.ultralytics_root).expanduser().resolve()
    pidnet_weights = Path(args.pidnet_weights).expanduser().resolve()
    fastsam_weights = Path(args.fastsam_weights).expanduser().resolve()
    device = resolve_torch_device(args.device)
    fastsam_device = resolve_ultralytics_device(device)

    setup_imports(pidnet_root, ultralytics_root)
    output_dirs = {
        "raw": output_root / "raw",
        "pidnet_semantic": output_root / "pidnet_semantic",
        "pidnet_road": output_root / "pidnet_road",
        "fastsam_all": output_root / "fastsam_all",
        "fastsam_text_road": output_root / "fastsam_text_road",
        "combined": output_root / "combined",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    pidnet = load_pidnet(pidnet_root, pidnet_weights, device)
    fastsam = load_fastsam(fastsam_weights, ultralytics_root)

    if args.warmup:
        warmup_path = image_root / SELECTED[0][1]
        warmup_image = cv2.imread(str(warmup_path), cv2.IMREAD_COLOR)
        if warmup_image is not None:
            pidnet_predict(pidnet, warmup_image, device)
            fastsam_predict(fastsam, warmup_path, warmup_image, args, fastsam_device)
            fastsam_predict(fastsam, warmup_path, warmup_image, args, fastsam_device, texts=["road"])

    rows = []
    for demo_id, image_name in SELECTED:
        image_path = image_root / image_name
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Missing image: {image_path}")

        cv2.imwrite(str(output_dirs["raw"] / f"{demo_id}_{image_name}"), image_bgr)

        _, pid_road, pid_semantic, pid_time = pidnet_predict(pidnet, image_bgr, device)
        pid_sem_overlay = overlay_mask(image_bgr, pid_semantic)
        pid_road_overlay = overlay_mask(image_bgr, pid_road, color=(0, 255, 0))

        fast_all, fast_all_vis, fast_all_count, fast_all_time = fastsam_predict(
            fastsam, image_path, image_bgr, args, fastsam_device
        )
        fast_road, fast_road_vis, fast_road_count, fast_road_time = fastsam_predict(
            fastsam, image_path, image_bgr, args, fastsam_device, texts=["road"]
        )
        fast_all_overlay = overlay_mask(image_bgr, fast_all_vis)
        fast_road_overlay = overlay_mask(image_bgr, fast_road, color=(0, 255, 0))

        boards = {
            "pidnet_semantic": build_triptych(
                image_bgr,
                pid_semantic,
                pid_sem_overlay,
                f"PIDNet | {image_name} | Cityscapes semantic parsing | no prompt",
            ),
            "pidnet_road": build_triptych(
                image_bgr,
                overlay_mask(np.zeros_like(image_bgr), pid_road, alpha=1.0, color=(0, 255, 0)),
                pid_road_overlay,
                f"PIDNet road | {image_name} | road class id = 0 | no prompt",
            ),
            "fastsam_all": build_triptych(
                image_bgr,
                fast_all_vis,
                fast_all_overlay,
                f"FastSAM | {image_name} | segment-all mode ({fast_all_count} masks)",
            ),
            "fastsam_text_road": build_triptych(
                image_bgr,
                overlay_mask(np.zeros_like(image_bgr), fast_road, alpha=1.0, color=(0, 255, 0)),
                fast_road_overlay,
                f"FastSAM | {image_name} | text prompt = road ({fast_road_count} masks)",
            ),
        }

        for key, board in boards.items():
            cv2.imwrite(str(output_dirs[key] / f"{demo_id}_{image_name}"), board)

        max_width = max(board.shape[1] for board in boards.values())
        padded = []
        for key in ["pidnet_semantic", "pidnet_road", "fastsam_all", "fastsam_text_road"]:
            board = boards[key]
            if board.shape[1] < max_width:
                pad = np.full((board.shape[0], max_width - board.shape[1], 3), 245, dtype=np.uint8)
                board = np.concatenate([board, pad], axis=1)
            padded.append(board)
        combined = np.concatenate(padded, axis=0)
        cv2.imwrite(str(output_dirs["combined"] / f"{demo_id}_{image_name}"), combined)

        for method, mask, mask_count, elapsed, prompt in [
            ("PIDNet semantic", pid_road, 1, pid_time, "none; semantic road class id 0"),
            ("FastSAM all", fast_all, fast_all_count, fast_all_time, "none; segment-all"),
            ("FastSAM text road", fast_road, fast_road_count, fast_road_time, "road"),
        ]:
            stats = mask_stats(mask)
            rows.append(
                {
                    "demo_id": demo_id,
                    "image_name": image_name,
                    "method": method,
                    "prompt": prompt,
                    "device": str(device),
                    "imgsz": "" if method.startswith("PIDNet") else args.fastsam_imgsz,
                    "conf": "" if method.startswith("PIDNet") else args.fastsam_conf,
                    "iou": "" if method.startswith("PIDNet") else args.fastsam_iou,
                    "inference_time_s": round(elapsed, 6),
                    "mask_count": mask_count,
                    "mask_pixels": stats["mask_pixels"],
                    "coverage_ratio": round(stats["coverage_ratio"], 6),
                    "component_count": stats["component_count"],
                    "largest_component_ratio": round(stats["largest_component_ratio"], 6),
                }
            )

    write_csv(output_root / "metrics.csv", rows)
    summary = {
        "device": str(device),
        "fastsam_device": str(fastsam_device),
        "num_images": len(SELECTED),
        "pidnet_weight": str(pidnet_weights),
        "fastsam_weight": str(fastsam_weights),
        "fastsam_params": {
            "imgsz": args.fastsam_imgsz,
            "conf": args.fastsam_conf,
            "iou": args.fastsam_iou,
        },
        "notes": [
            "No pixel-level road ground truth is available; metrics are proxy metrics, not accuracy.",
            "PIDNet has no prompt and uses Cityscapes semantic road class id 0.",
            "FastSAM all mode has no prompt and segments all candidate instances.",
            "FastSAM text-road mode uses text prompt 'road' and may require CLIP.",
        ],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "metrics": str(output_root / "metrics.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
