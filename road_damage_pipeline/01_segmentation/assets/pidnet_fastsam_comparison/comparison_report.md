# PIDNet vs FastSAM Visual Comparison

## Image Set

The 12 images are selected from `RDD2022/RDD_SPLIT/test/images`.

| Demo ID | Image |
| --- | --- |
| D002 | `Czech_000201.jpg` |
| D005 | `Czech_000194.jpg` |
| D008 | `India_000090.jpg` |
| D011 | `India_000118.jpg` |
| D014 | `Japan_000035.jpg` |
| D017 | `Japan_000076.jpg` |
| D020 | `Norway_000031.jpg` |
| D023 | `Norway_000059.jpg` |
| D026 | `United_States_000046.jpg` |
| D029 | `United_States_000083.jpg` |
| D032 | `China_MotorBike_000036.jpg` |
| D035 | `China_MotorBike_000160.jpg` |

## Methods

| Method | Prompt | Output Meaning |
| --- | --- | --- |
| PIDNet semantic | None | Cityscapes 19-class semantic parsing; road is class id `0`. |
| FastSAM all | None | Segment all candidate instances in the image. |
| FastSAM text road | `road` | Use CLIP text filtering to select masks related to `road`. |

PIDNet does not need a prompt. FastSAM all mode also has no prompt. FastSAM text-road mode uses the prompt `road`.

## Proxy Metrics

There is no pixel-level road mask ground truth in RDD2022, so these are proxy metrics, not segmentation accuracy.

| Metric | Meaning |
| --- | --- |
| `median_time_s` | More stable inference-time indicator on MPS because first-run and shape-specific compilation can skew the mean. |
| `mean_time_s` | Average measured inference time over the 12 images. |
| `mean_mask_count` | Number of predicted masks. For road preprocessing, very high values usually mean the method is segmenting many unrelated instances. |
| `mean_coverage_ratio` | Fraction of image pixels included in the selected mask union. |
| `mean_component_count` | Number of connected components in the selected mask union. |
| `mean_largest_component_ratio` | Largest connected component divided by total selected mask pixels. Higher means a more continuous selected region. |

## Summary Results

Device: `mps`.

FastSAM parameters: `imgsz=768`, `conf=0.1`, `iou=0.9`.

| Method | Mean Time (s) | Median Time (s) | Mean Mask Count | Mean Coverage | Mean Components | Largest Component Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PIDNet semantic | 1.0966 | 0.1733 | 1.00 | 0.3999 | 11.25 | 0.9869 |
| FastSAM all | 0.5435 | 0.4593 | 122.33 | 0.8180 | 8.75 | 0.9794 |
| FastSAM text road | 2.2587 | 1.6567 | 1.00 | 0.3156 | 5.17 | 0.9892 |

## Resource Notes

| Method | Local Checkpoint | Extra Dependency |
| --- | ---: | --- |
| PIDNet-L Cityscapes | 143 MB | None beyond PyTorch/OpenCV. |
| FastSAM-s all mode | 23 MB | Ultralytics FastSAM. |
| FastSAM-s text-road | 23 MB | Uses CLIP branch; first run may download about 338 MB of CLIP weights. |

## Evidence-Based Takeaways

- If the criterion is only checkpoint size, FastSAM-s is smaller than PIDNet-L.
- If the criterion is actual road-region preprocessing, FastSAM all mode is not suitable by itself because it produces many masks and covers most of the image, including non-road objects.
- FastSAM text-road is closer to road extraction than all mode, but it is slower and depends on the CLIP text branch.
- PIDNet directly predicts the road semantic class without prompt engineering. Its median inference time is lower than both FastSAM modes in this 12-image MPS test.
- The existing PIDRoad preprocessing still has a limitation: it preserves GT bbox regions to avoid cutting damage pixels, so it is a controlled dataset experiment rather than a pure deployment pipeline.

## Output Files

| Output | Path |
| --- | --- |
| Per-image combined boards | `segmentation_pipeline/outputs/pidnet_fastsam_comparison/combined/` |
| PIDNet semantic boards | `segmentation_pipeline/outputs/pidnet_fastsam_comparison/pidnet_semantic/` |
| PIDNet road-mask boards | `segmentation_pipeline/outputs/pidnet_fastsam_comparison/pidnet_road/` |
| FastSAM all-mode boards | `segmentation_pipeline/outputs/pidnet_fastsam_comparison/fastsam_all/` |
| FastSAM text-road boards | `segmentation_pipeline/outputs/pidnet_fastsam_comparison/fastsam_text_road/` |
| Per-image metrics | `segmentation_pipeline/outputs/pidnet_fastsam_comparison/metrics.csv` |
| Method summary | `segmentation_pipeline/outputs/pidnet_fastsam_comparison/method_summary.csv` |
