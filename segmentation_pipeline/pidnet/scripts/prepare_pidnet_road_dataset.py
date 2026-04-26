import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
PIDNET_ROOT = PIPELINE_ROOT / 'pidnet' / 'source'
PIDNET_WEIGHTS = PIPELINE_ROOT / 'pidnet' / 'weights' / 'PIDNet_L_Cityscapes_test.pt'
INPUT_ROOT = PROJECT_ROOT / 'RDD2022' / 'RDD_SPLIT'
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / 'outputs' / 'RDD_SPLIT_pidroad_nondrone'
DEFAULT_YAML_PATH = PIPELINE_ROOT / 'outputs' / 'rdd2022_pidroad_nondrone.yaml'

PIDNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIDNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = {
    0: 'Longitudinal Crack',
    1: 'Transverse Crack',
    2: 'Alligator Crack',
    3: 'Pothole',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Build a non-drone road-masked RDD2022 dataset using PIDNet.')
    parser.add_argument('--input-root', default=str(INPUT_ROOT), help='Original RDD2022 split root.')
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT), help='Output dataset root.')
    parser.add_argument('--yaml-out', default=str(DEFAULT_YAML_PATH), help='Path to write the generated dataset yaml.')
    parser.add_argument('--pidnet-root', default=str(PIDNET_ROOT), help='PIDNet source root.')
    parser.add_argument('--weights', default=str(PIDNET_WEIGHTS), help='PIDNet Cityscapes checkpoint.')
    parser.add_argument('--splits', nargs='*', default=['train', 'val'], help='Dataset splits to process.')
    parser.add_argument('--include-test', action='store_true', help='Also process the test split.')
    parser.add_argument('--exclude-drone', action='store_true', default=True, help='Exclude files whose names contain Drone.')
    parser.add_argument('--road-class-id', type=int, default=0, help='PIDNet class id used for road in Cityscapes.')
    parser.add_argument('--road-dilate', type=int, default=15, help='Odd kernel size used to dilate the road mask.')
    parser.add_argument('--bbox-pad-px', type=int, default=8, help='Minimum bbox padding in pixels to preserve context.')
    parser.add_argument('--bbox-pad-ratio', type=float, default=0.02, help='Additional bbox padding as a fraction of box size.')
    parser.add_argument('--limit', type=int, default=0, help='Optional max number of images per split for quick testing.')
    parser.add_argument('--preview-count', type=int, default=8, help='How many preview boards to save per split.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip samples whose output image already exists.')
    return parser.parse_args()


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ensure_pidnet_importable(pidnet_root: Path):
    if str(pidnet_root) not in sys.path:
        sys.path.insert(0, str(pidnet_root))


def load_pidnet_model(pidnet_root: Path, weights_path: Path):
    ensure_pidnet_importable(pidnet_root)
    import models

    model = models.pidnet.get_pred_model('pidnet-l', 19)
    pretrained = torch.load(weights_path, map_location='cpu')
    pretrained = pretrained['state_dict'] if 'state_dict' in pretrained else pretrained
    model_dict = model.state_dict()
    pretrained = {
        key[6:]: value
        for key, value in pretrained.items()
        if key.startswith('model.') and key[6:] in model_dict and model_dict[key[6:]].shape == value.shape
    }
    model_dict.update(pretrained)
    model.load_state_dict(model_dict, strict=False)
    model = model.to(get_device())
    model.eval()
    return model


def pidnet_input_transform(image_bgr: np.ndarray):
    image = image_bgr.astype(np.float32)[:, :, ::-1] / 255.0
    image -= PIDNET_MEAN
    image /= PIDNET_STD
    return image.transpose((2, 0, 1)).copy()


def pidnet_predict(model, image_bgr: np.ndarray):
    tensor = torch.from_numpy(pidnet_input_transform(image_bgr)).unsqueeze(0).to(get_device())
    height, width = tensor.shape[-2:]
    aligned_height = int(np.ceil(height / 32.0) * 32)
    aligned_width = int(np.ceil(width / 32.0) * 32)
    if (aligned_height, aligned_width) != (height, width):
        tensor = F.interpolate(tensor, size=(aligned_height, aligned_width), mode='bilinear', align_corners=True)
    with torch.no_grad():
        pred = model(tensor)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def load_yolo_boxes(label_path: Path, width: int, height: int, pad_px: int, pad_ratio: float):
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        box_w = bw * width
        box_h = bh * height
        center_x = cx * width
        center_y = cy * height
        pad_x = max(pad_px, int(round(box_w * pad_ratio)))
        pad_y = max(pad_px, int(round(box_h * pad_ratio)))
        x1 = max(0, int(round(center_x - box_w / 2.0)) - pad_x)
        y1 = max(0, int(round(center_y - box_h / 2.0)) - pad_y)
        x2 = min(width, int(round(center_x + box_w / 2.0)) + pad_x)
        y2 = min(height, int(round(center_y + box_h / 2.0)) + pad_y)
        boxes.append((x1, y1, x2, y2))
    return boxes


def build_keep_mask(road_pred: np.ndarray, road_class_id: int, road_dilate: int, boxes):
    keep_mask = (road_pred == road_class_id).astype(np.uint8)
    if road_dilate > 1:
        kernel_size = road_dilate if road_dilate % 2 == 1 else road_dilate + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        keep_mask = cv2.dilate(keep_mask, kernel, iterations=1)
    for x1, y1, x2, y2 in boxes:
        keep_mask[y1:y2, x1:x2] = 1
    return keep_mask


def apply_mask(image_bgr: np.ndarray, keep_mask: np.ndarray):
    masked = np.zeros_like(image_bgr)
    masked[keep_mask > 0] = image_bgr[keep_mask > 0]
    return masked


def overlay_mask(image_bgr: np.ndarray, keep_mask: np.ndarray):
    overlay = image_bgr.copy()
    tint = np.zeros_like(image_bgr)
    tint[:, :, 1] = 180
    road = keep_mask > 0
    overlay[road] = cv2.addWeighted(image_bgr[road], 0.6, tint[road], 0.4, 0.0)
    return overlay


def draw_boxes(image_bgr: np.ndarray, boxes):
    image = image_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
    return image


def make_preview(image_bgr: np.ndarray, road_mask: np.ndarray, keep_mask: np.ndarray, boxes, masked_bgr: np.ndarray):
    road_vis = np.zeros_like(image_bgr)
    road_vis[road_mask > 0] = (0, 255, 0)
    keep_vis = np.zeros_like(image_bgr)
    keep_vis[keep_mask > 0] = (255, 255, 255)

    original = draw_boxes(image_bgr, boxes)
    overlay = draw_boxes(overlay_mask(image_bgr, keep_mask), boxes)
    masked = draw_boxes(masked_bgr, boxes)
    road_vis = draw_boxes(road_vis, boxes)
    keep_vis = draw_boxes(keep_vis, boxes)

    panels = [original, road_vis, keep_vis, overlay, masked]
    titled = []
    for title, panel in zip(['Original', 'Road', 'Keep', 'Overlay', 'Masked'], panels):
        canvas = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        titled.append(canvas)
    return np.concatenate(titled, axis=1)


def iter_split_images(split_root: Path, exclude_drone: bool):
    image_dir = split_root / 'images'
    for image_path in sorted(image_dir.glob('*.jpg')):
        if exclude_drone and 'Drone' in image_path.name:
            continue
        yield image_path


def process_split(model, split: str, args, input_root: Path, output_root: Path):
    split_root = input_root / split
    out_image_dir = output_root / split / 'images'
    out_label_dir = output_root / split / 'labels'
    preview_dir = output_root / 'previews' / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    preview_saved = 0
    skipped_existing = 0
    excluded_drone = 0

    all_images = sorted((split_root / 'images').glob('*.jpg'))
    if args.exclude_drone:
        excluded_drone = sum('Drone' in path.name for path in all_images)

    for image_index, image_path in enumerate(iter_split_images(split_root, args.exclude_drone), start=1):
        if args.limit and processed >= args.limit:
            break

        out_image_path = out_image_dir / image_path.name
        out_label_path = out_label_dir / f'{image_path.stem}.txt'
        label_path = split_root / 'labels' / f'{image_path.stem}.txt'

        if args.skip_existing and out_image_path.is_file() and out_label_path.is_file():
            skipped_existing += 1
            processed += 1
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        height, width = image_bgr.shape[:2]
        road_pred = pidnet_predict(model, image_bgr)
        boxes = load_yolo_boxes(label_path, width, height, args.bbox_pad_px, args.bbox_pad_ratio)
        keep_mask = build_keep_mask(road_pred, args.road_class_id, args.road_dilate, boxes)
        masked_bgr = apply_mask(image_bgr, keep_mask)

        cv2.imwrite(str(out_image_path), masked_bgr)
        if label_path.is_file():
            shutil.copy2(label_path, out_label_path)

        if preview_saved < args.preview_count:
            road_mask = (road_pred == args.road_class_id).astype(np.uint8)
            preview = make_preview(image_bgr, road_mask, keep_mask, boxes, masked_bgr)
            cv2.imwrite(str(preview_dir / f'{image_path.stem}_preview.jpg'), preview)
            preview_saved += 1

        processed += 1
        if processed % 100 == 0:
            print(json.dumps({'split': split, 'processed': processed, 'last_image': image_path.name}, ensure_ascii=False))

    return {
        'split': split,
        'processed': processed,
        'excluded_drone': excluded_drone,
        'skipped_existing': skipped_existing,
        'preview_saved': preview_saved,
    }


def write_dataset_yaml(output_root: Path, yaml_out: Path, include_test: bool):
    lines = [
        '# RDD2022 non-drone road-masked dataset built with PIDNet',
        f'path: {output_root}',
        'train: train/images',
        'val: val/images',
    ]
    if include_test:
        lines.append('test: test/images')
    lines.extend([
        'names:',
        '  0: Longitudinal Crack',
        '  1: Transverse Crack',
        '  2: Alligator Crack',
        '  3: Pothole',
        '',
    ])
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text('\n'.join(lines), encoding='utf-8')


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    yaml_out = Path(args.yaml_out).expanduser().resolve()
    pidnet_root = Path(args.pidnet_root).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()

    splits = list(args.splits)
    if args.include_test and 'test' not in splits:
        splits.append('test')

    print('=== Build PIDNet road-masked RDD2022 dataset ===')
    print(f'Input root: {input_root}')
    print(f'Output root: {output_root}')
    print(f'Splits: {splits}')
    print(f'Exclude drone: {args.exclude_drone}')
    print(f'Road class id: {args.road_class_id}')
    print(f'PIDNet root: {pidnet_root}')
    print(f'PIDNet weights: {weights_path}')

    model = load_pidnet_model(pidnet_root, weights_path)

    stats = []
    for split in splits:
        stats.append(process_split(model, split, args, input_root, output_root))

    write_dataset_yaml(output_root, yaml_out, 'test' in splits)

    stats_path = output_root / 'build_stats.json'
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'yaml': str(yaml_out), 'stats': stats}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
