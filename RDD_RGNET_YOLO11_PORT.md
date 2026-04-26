# RDD-RGNet to YOLO11 Port

This port adds the minimum changes needed to reproduce the core idea of RDD-RGNet on top of Ultralytics YOLO11.

## Added components

- `ultralytics/nn/modules/rdd_rg.py`
  - `CoordAtt`
  - `binary_spatial_Attention`
- `ultralytics/utils/generate_mask.py`
  - weak region masks from detection boxes
- `ultralytics/cfg/models/11/yolo11-rg.yaml`
  - YOLO11 backbone + First-RG + Second-RG style head

## Patched components

- `ultralytics/nn/tasks.py`
  - module registration
  - custom two-stage forward path
  - custom loss return `(loss, loss_items, mask_pred_quarter)`
- `ultralytics/engine/trainer.py`
  - weak-mask generation during training
  - region loss added to detection loss
- `ultralytics/models/yolo/detect/val.py`
  - tuple output compatibility during validation

## Important note

This is a manual port based on the published YOLOv8 RDD-RGNet code. The original repository does not publish its YOLO11 implementation, so this YOLO11 graph is an engineering adaptation rather than an official upstream file.

## How the mask supervision works

For each image, every YOLO-format bbox is rasterized into a white rectangle on a black background. That binary image is used as a weak region mask target.

## Example

```bash
cd ultralytics_yolo11_final
pip install -e .
python examples/rdd_rgnet/run_train_yolo11_rg.py
```

The example script uses paths relative to `ultralytics_yolo11_final`. Set `YOLO_MODEL_CFG` or `YOLO_PRETRAINED_WEIGHTS` if you need to override the defaults.
