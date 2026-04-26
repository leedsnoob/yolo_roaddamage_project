# Unified RDD YOLO11 Fork

This fork is the local retest target for the thesis experiments. It is based on
`ultralytics_yolo11_rg11_sppf_lska_strip_wiou/` and adds compatibility for the
other historical forks used in `runs1-4`.

Supported custom modules:

- RG11 region guidance: `binary_spatial_Attention`, `CoordAtt`, `DWSConvDown`
- DCNv2 + BiFPN: `DCNv2`, `BiFPN_Fuse`
- Strip/WIoU branches: `StripConvAttention`, YAML-level `wiou: true`
- P2/DySample branch: `DySample`
- SPPF-LSKA branch: `LSKblock`, `SPPF_LSKA`
- Exploratory LSKA/DySample/ShapeIoU branch: `DACA`, `C2_DACA`

Compatibility policy:

- Original `runs/**/args.yaml` files are treated as immutable evidence.
- Server-side paths are mapped by `thesis_package/engineering/check_runs_module_compatibility.py`.
- Old YOLO forks must not be deleted until `runs_final/local_retest/runs_module_compatibility.csv` confirms the important YAML and weight checks.

Important note:

The dynamic-region-loss fork currently has the same WIoU loss file hash as the
plain Strip+WIoU fork. Do not describe it as a separate code module unless future
inspection finds an actual code-level difference.
