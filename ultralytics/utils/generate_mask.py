import numpy as np
import torch


def generate_mask(batch: dict) -> torch.Tensor:
    """Rasterize YOLO-format boxes into weak binary masks."""
    num_img = batch["img"].shape[0]
    mask_batch = torch.zeros((num_img, 1, batch["img"].shape[2], batch["img"].shape[3]))
    batch_idx = batch["batch_idx"].tolist()

    for i in range(num_img):
        labels_idx = [m for m, n in enumerate(batch_idx) if n == i]
        labels = batch["bboxes"][labels_idx].cpu().numpy()
        height, width = batch["img"][i].shape[1:]
        mask = np.zeros((1, height, width), dtype=np.float32)

        for cx, cy, bw, bh in labels:
            left = max(int((cx - bw / 2) * width + 0.5), 0)
            top = max(int((cy - bh / 2) * height + 0.5), 0)
            right = min(int((cx + bw / 2) * width + 0.5), width)
            bottom = min(int((cy + bh / 2) * height + 0.5), height)
            mask[0, top:bottom, left:right] = 255

        mask_batch[i] = torch.from_numpy(mask)

    return mask_batch
