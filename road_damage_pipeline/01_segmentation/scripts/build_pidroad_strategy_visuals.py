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
IMAGE_ROOT = PROJECT_ROOT / "RDD2022" / "RDD_SPLIT" / "test" / "images"
LABEL_ROOT = PROJECT_ROOT / "RDD2022" / "RDD_SPLIT" / "test" / "labels"
OUTPUT_ROOT = PIPELINE_ROOT / "outputs" / "pidroad_strategy_visuals"

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
CLASS_NAMES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
}
CLASS_COLORS = {
    0: (245, 158, 11),
    1: (59, 130, 246),
    2: (16, 185, 129),
    3: (239, 68, 68),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PIDRoad preprocessing strategy.")
    parser.add_argument("--image-root", default=str(IMAGE_ROOT))
    parser.add_argument("--label-root", default=str(LABEL_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--pidnet-root", default=str(PIDNET_ROOT))
    parser.add_argument("--weights", default=str(PIDNET_WEIGHTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--road-class-id", type=int, default=0)
    parser.add_argument("--road-dilate", type=int, default=15)
    parser.add_argument("--bbox-pad-px", type=int, default=8)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.02)
    return parser.parse_args()


def resolve_device(device: str):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_imports(pidnet_root: Path):
    if str(pidnet_root) not in sys.path:
        sys.path.insert(0, str(pidnet_root))


def load_pidnet(pidnet_root: Path, weights_path: Path, device: torch.device):
    setup_imports(pidnet_root)
    import models

    model = models.pidnet.get_pred_model("pidnet-l", 19)
    pretrained = torch.load(weights_path, map_location="cpu")
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
    return pred, time.perf_counter() - start


def read_yolo_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, bw, bh = parts
        boxes.append((int(float(cls_id)), float(cx), float(cy), float(bw), float(bh)))
    return boxes


def yolo_to_xyxy(labels, width: int, height: int, pad_px=0, pad_ratio=0.0):
    boxes = []
    for cls_id, cx, cy, bw, bh in labels:
        box_w = bw * width
        box_h = bh * height
        center_x = cx * width
        center_y = cy * height
        pad_x = max(pad_px, int(round(box_w * pad_ratio))) if (pad_px or pad_ratio) else 0
        pad_y = max(pad_px, int(round(box_h * pad_ratio))) if (pad_px or pad_ratio) else 0
        x1 = max(0, int(round(center_x - box_w / 2.0)) - pad_x)
        y1 = max(0, int(round(center_y - box_h / 2.0)) - pad_y)
        x2 = min(width, int(round(center_x + box_w / 2.0)) + pad_x)
        y2 = min(height, int(round(center_y + box_h / 2.0)) + pad_y)
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def draw_boxes(image_bgr: np.ndarray, boxes):
    canvas = image_bgr.copy()
    for cls_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        text = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        y_top = max(0, y1 - th - 8)
        cv2.rectangle(canvas, (x1, y_top), (x1 + tw + 10, y_top + th + 8), color, -1)
        cv2.putText(canvas, text, (x1 + 5, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def build_keep_mask(road_mask: np.ndarray, boxes, road_dilate: int):
    keep_mask = road_mask.astype(np.uint8).copy()
    if road_dilate > 1:
        kernel_size = road_dilate if road_dilate % 2 == 1 else road_dilate + 1
        keep_mask = cv2.dilate(keep_mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    for _, x1, y1, x2, y2 in boxes:
        keep_mask[y1:y2, x1:x2] = 1
    return keep_mask


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray):
    out = np.zeros_like(image_bgr)
    out[mask > 0] = image_bgr[mask > 0]
    return out


def add_caption(image_bgr: np.ndarray, caption: str):
    bar = np.full((44, image_bgr.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, caption, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([image_bgr, bar])


def make_triptych(images):
    height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] == height:
            padded.append(image)
            continue
        pad = np.full((height - image.shape[0], image.shape[1], 3), 245, dtype=np.uint8)
        padded.append(np.vstack([image, pad]))
    separator = np.full((height, 10, 3), 245, dtype=np.uint8)
    canvas = padded[0]
    for image in padded[1:]:
        canvas = np.hstack([canvas, separator, image])
    return canvas


def mask_stats(mask: np.ndarray):
    pixels = int((mask > 0).sum())
    return pixels, pixels / int(mask.size)


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    image_root = Path(args.image_root).expanduser().resolve()
    label_root = Path(args.label_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    pidnet_root = Path(args.pidnet_root).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    device = resolve_device(args.device)

    dirs = {
        "01_original_bbox": output_root / "01_original_bbox",
        "02_pidnet_road": output_root / "02_pidnet_road",
        "03_pidnet_road_bbox_preserve": output_root / "03_pidnet_road_bbox_preserve",
        "04_triptych": output_root / "04_triptych",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    model = load_pidnet(pidnet_root, weights, device)
    rows = []

    for demo_id, image_name in SELECTED:
        image_path = image_root / image_name
        label_path = label_root / f"{Path(image_name).stem}.txt"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        height, width = image_bgr.shape[:2]
        labels = read_yolo_labels(label_path)
        boxes = yolo_to_xyxy(labels, width, height)
        padded_boxes = yolo_to_xyxy(labels, width, height, args.bbox_pad_px, args.bbox_pad_ratio)

        pred, elapsed = pidnet_predict(model, image_bgr, device)
        road_mask = (pred == args.road_class_id).astype(np.uint8)
        keep_mask = build_keep_mask(road_mask, padded_boxes, args.road_dilate)

        original_bbox = add_caption(
            draw_boxes(image_bgr, boxes),
            f"1 Original + GT bbox | {image_name}",
        )
        pidnet_road = add_caption(
            apply_mask(image_bgr, road_mask),
            f"2 PIDNet road only | road class id = {args.road_class_id}",
        )
        pidnet_keep = add_caption(
            draw_boxes(apply_mask(image_bgr, keep_mask), boxes),
            "3 PIDNet road + bbox preserve",
        )

        cv2.imwrite(str(dirs["01_original_bbox"] / f"{demo_id}_{image_name}"), original_bbox)
        cv2.imwrite(str(dirs["02_pidnet_road"] / f"{demo_id}_{image_name}"), pidnet_road)
        cv2.imwrite(str(dirs["03_pidnet_road_bbox_preserve"] / f"{demo_id}_{image_name}"), pidnet_keep)
        cv2.imwrite(
            str(dirs["04_triptych"] / f"{demo_id}_{image_name}"),
            make_triptych([original_bbox, pidnet_road, pidnet_keep]),
        )

        road_pixels, road_ratio = mask_stats(road_mask)
        keep_pixels, keep_ratio = mask_stats(keep_mask)
        rows.append(
            {
                "demo_id": demo_id,
                "image_name": image_name,
                "label_count": len(labels),
                "device": str(device),
                "pidnet_time_s": round(elapsed, 6),
                "road_pixels": road_pixels,
                "road_ratio": round(road_ratio, 6),
                "keep_pixels": keep_pixels,
                "keep_ratio": round(keep_ratio, 6),
                "added_by_bbox_pixels": int(max(0, keep_pixels - road_pixels)),
                "road_dilate": args.road_dilate,
                "bbox_pad_px": args.bbox_pad_px,
                "bbox_pad_ratio": args.bbox_pad_ratio,
            }
        )

    write_csv(output_root / "strategy_metrics.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "num_images": len(SELECTED),
                "outputs": {key: str(path) for key, path in dirs.items()},
                "notes": [
                    "Each sample is exported as three separate images and one per-sample triptych.",
                    "The script does not create a single board that combines all samples.",
                    "PIDNet road-only uses Cityscapes road class id 0.",
                    "PIDNet road + bbox preserve applies road dilation and preserves GT bbox regions.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "metrics": str(output_root / "strategy_metrics.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
