# Area Measurement Pipeline V1

This folder combines the existing area-estimation experiments into one pipeline artifact.
The input is the selected RDD test images and their GT bounding boxes from txt labels.

## Methods

- M1 bbox geometry prior: class-specific bbox formula with fixed pixel scale.
- M2 FastSAM mask: GT bbox is cropped first; FastSAM receives the crop and a class-derived damage text prompt.
- M3 Depth Anything V2 + bbox geometry prior: bbox depth area corrected by the class-specific effective-area ratio.
- M4 Metric3D + bbox geometry prior: bbox depth area corrected by the class-specific effective-area ratio.

## Parameters

- scale_factor_m_per_px: `0.01`
- assumed_horizontal_fov_deg: `70.0`
- fastsam_model: `FastSAM-s.pt`
- Depth Anything V2 model: `Depth-Anything-V2-Metric-VKITTI-Small`
- Metric3D model: `metric3d_vit_small`
- Metric3D input size: `[616, 1064]`

## Outputs

- `four_method_area_results_long.csv`: one row per bbox per method.
- `four_method_area_results_wide.csv`: one row per bbox with all four method values.
- `four_method_area_summary_by_class.csv`: per-class summary.
- `visuals/four_method_summary_table.png`: compact table for paper/frontend.
- `visuals/four_method_boards/`: per-image boards showing bbox prior, FastSAM crop mask, Depth Anything V2, and Metric3D.

## Main Caveat

No camera intrinsics or true physical labels are available. These are estimated areas for method comparison, not ground-truth measurements.

## Counts

- GT boxes: `11`
- method rows: `44`
- generated per-image boards: `10`

## Quick Result Table

| class | method | mean area m2 | median area m2 | fallback | assumption |
|---|---|---:|---:|---:|---:|
| D00 | M1 Empirical bbox | 3.493333 | 1.656000 | 0 | 0 |
| D00 | M2 FastSAM mask | 8.846700 | 0.985800 | 0 | 0 |
| D00 | M3 Depth Anything V2 | 12.743254 | 13.445993 | 0 | 3 |
| D00 | M4 Metric3D | 2.208838 | 2.134403 | 0 | 3 |
| D10 | M1 Empirical bbox | 10.720000 | 4.668000 | 0 | 0 |
| D10 | M2 FastSAM mask | 9.055700 | 3.617700 | 2 | 0 |
| D10 | M3 Depth Anything V2 | 25.415319 | 32.785323 | 0 | 3 |
| D10 | M4 Metric3D | 3.825247 | 4.687925 | 0 | 3 |
| D20 | M1 Empirical bbox | 2.758400 | 2.758400 | 0 | 0 |
| D20 | M2 FastSAM mask | 1.428400 | 1.428400 | 0 | 0 |
| D20 | M3 Depth Anything V2 | 31.284404 | 31.284404 | 0 | 2 |
| D20 | M4 Metric3D | 5.060378 | 5.060378 | 0 | 2 |
| D40 | M1 Empirical bbox | 1.044922 | 0.270600 | 0 | 0 |
| D40 | M2 FastSAM mask | 1.624333 | 0.687400 | 0 | 0 |
| D40 | M3 Depth Anything V2 | 26.432468 | 8.427678 | 0 | 3 |
| D40 | M4 Metric3D | 3.871831 | 1.474417 | 0 | 3 |
