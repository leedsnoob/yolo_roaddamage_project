# Evidence-Grounded Road Damage Report Agent

## 1. Why This Module Is Needed

The proposal defines the final system as a five-stage workflow: segmentation, detection, tracking/deduplication, area estimation, and report generation. The report module should not be treated as a standalone text generator. It should be the explanation layer on top of the upstream evidence.

The key risk is hallucination. A generic VLM can describe images fluently, but road damage reporting needs exact class names, event counts, time ranges, estimated areas, and limitations. Therefore, the report agent should be evidence-grounded: every number and every damage event in the report must come from CSV/JSON outputs produced by earlier modules.

## 2. What DamageQwen Contributes

DamageQwen is directly relevant, but its scope is narrower than our full pipeline.

DamageQwen workflow:

1. YOLOv11x detects candidate pavement defects.
2. CLIP removes duplicate detections that NMS may miss, especially same-frame duplicate boxes or cross-class duplicate boxes.
3. The refined class and coordinate information is converted into a structured prompt.
4. Qwen2-VL generates natural-language pavement damage descriptions.

What we should borrow:

- Use detector outputs as explicit textual grounding.
- Provide coordinates, class names, and geometric measurements in the prompt.
- Use few-shot examples with expert-style descriptions.
- Evaluate generated reports with both text metrics and domain-specific checking.

What we should not copy directly:

- DamageQwen focuses mainly on single-image descriptions; our system needs video-level event reports.
- Its CLIP duplicate removal is same-frame duplicate removal, not cross-frame video deduplication.
- Its measurements are pixel-level or approximate; our area module must explicitly state physical-scale assumptions.

## 3. Related Intelligent Report / Agent Ideas

### Medical report generation

Radiology report generation work such as R2Gen shows that professional report generation needs image evidence, domain language, and structured memory, not only generic image captioning. However, medical report papers also show that BLEU/ROUGE alone cannot prove report correctness. For road inspection, we should add rule-based factual checks and human review.

### Multimodal tool agents

MM-REACT, Visual ChatGPT, HuggingGPT, and LLaVA-Plus all follow a similar principle: the language model should orchestrate specialized visual tools instead of solving every visual task itself. This fits our project because detection, deduplication, and area estimation are already specialized tools.

### Construction reporting

AutoRepo is relevant because it treats report generation as a complete inspection workflow: data acquisition, multimodal analysis, report generation, and human review. The useful idea is not only "generate text", but "generate a traceable report from inspection evidence".

## 4. Final Architecture

The recommended architecture is:

```text
Pipeline outputs
  -> Evidence Builder
  -> Visual Evidence Selector
  -> Report Planner Agent
  -> Evidence-Grounded Writer Agent
  -> Verifier Agent
  -> Markdown / HTML / PDF export
```

### Evidence Builder

Input files:

- detection metrics from `02_detection`;
- per-frame detections and model metadata;
- `track_events.csv` and `summary.json` from `03_video_dedup`;
- area result tables and visual boards from `04_area_measurement`;
- selected keyframes and annotated images.

Output:

- `report_input.json`, following `schemas/report_input.schema.json`.

### Visual Evidence Selector

Selects representative evidence:

- one overview frame;
- 3 to 5 high-confidence damage events;
- dense damage segment if video is used;
- area visual boards for selected images.

The selector should prefer diverse damage classes and avoid showing near-identical frames.

### Report Planner Agent

Creates a fixed report outline:

1. Inspection overview.
2. Damage statistics.
3. Event-level damage list.
4. Area estimation summary.
5. Priority and maintenance suggestion.
6. Evidence figures.
7. Limitations.

### Evidence-Grounded Writer Agent

Writes the report from `report_input.json` and selected images. It must not create new damage events, new numerical values, or unsupported severity claims.

### Verifier Agent

Checks factual consistency:

- every count appears in JSON/CSV;
- every class name is one of `D00`, `D10`, `D20`, `D40`;
- every area value is marked as estimated;
- confidence is not called accuracy;
- no claim of physical GT area appears without calibration evidence;
- maintenance advice is phrased as decision support.

## 5. Report Data Contract

The agent should receive structured data before images.

Minimum fields:

- `inspection_id`
- `source`
- `model`
- `dataset_protocol`
- `damage_summary`
- `events`
- `area_estimates`
- `visual_evidence`
- `limitations`

This makes the generated report auditable and easy to connect to a web frontend.

## 6. Recommended Implementation Route

### Version 1: no fine-tuning

Use Qwen2-VL / Qwen2.5-VL / Qwen-VL-Max or another VLM with:

- structured evidence JSON;
- 2 to 4 few-shot examples;
- selected keyframes;
- strict verifier prompt.

This version is enough for the thesis demo because it is controllable and can be evaluated.

### Version 2: few-shot road inspection examples

Manually write 5 to 10 report examples:

- one light damage case;
- one dense damage case;
- one false-positive/uncertain case;
- one area-estimation case;
- one video deduplication case.

Use them as in-context examples.

### Version 3: LoRA fine-tuning

Only consider LoRA after collecting enough expert reports. For a thesis project, fewer than several hundred high-quality reports is usually not enough to justify LoRA as the main method.

## 7. Evaluation Plan

Use three levels of evaluation.

### Factual consistency

Check automatically:

- event counts match `track_events.csv`;
- class counts match `summary.json`;
- area values match area CSV;
- no unsupported class or measurement appears.

### Text quality

If reference reports are available, compute:

- ROUGE;
- BLEU;
- METEOR;
- BERTScore.

These metrics can be reported as auxiliary metrics, following DamageQwen, but they are not enough by themselves.

### Domain review

Ask a human reviewer to score:

- correctness;
- completeness;
- clarity;
- usefulness of maintenance suggestion;
- whether uncertainty is properly stated.

## 8. Thesis Innovation

The strongest claim should not be "we fine-tuned a VLM". The stronger and safer claim is:

> We extend detector-enhanced VLM reporting from single-image pavement description to a traceable video-level road inspection report, integrating detection, cross-frame event deduplication, and assumption-aware area estimation.

This is more aligned with the actual work and avoids unsupported model-training claims.

## 9. Practical Prompt Rule

The report writer must follow this rule:

> If a fact is not in `report_input.json`, do not write it as a fact.

This is the main difference between a useful inspection report agent and a generic image captioning demo.

## 10. Source Papers Checked

- DamageQwen: Zhang and Liu, "Vision-enhanced multi-modal learning framework for non-destructive pavement damage detection", Automation in Construction, 2025. https://www.sciencedirect.com/science/article/pii/S0926580525004297
- Qwen2-VL: Wang et al., "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution", 2024. https://arxiv.org/abs/2409.12191
- AutoRepo: Pu et al., "AutoRepo: A general framework for multimodal LLM-based automated construction reporting", Expert Systems with Applications, 2024. https://www.sciencedirect.com/science/article/pii/S0957417424014684
- R2Gen: Chen et al., "Generating Radiology Reports via Memory-driven Transformer", EMNLP 2020. https://aclanthology.org/2020.emnlp-main.112/
- MM-REACT: Yang et al., "MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action", 2023. https://arxiv.org/abs/2303.11381
- Visual ChatGPT: Wu et al., "Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models", 2023. https://arxiv.org/abs/2303.04671
- LLaVA-Plus: Liu et al., "LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents", 2023. https://arxiv.org/abs/2311.05437
