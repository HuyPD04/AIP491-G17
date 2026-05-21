# Data Analysis and YOLO Baseline for Adaptive ROI-SAHI

This folder is independent from the older project code. It reads the existing VisDrone data from:

- `../../data/raw`
- `../../data/processed`

It does not modify the old `src`, `scripts`, `tests`, or config files.

## Phase Coverage

- Dataset validation and statistics
- Object size, spatial, density, and border EDA
- YOLO validation baseline
- Small-object failure and confidence analysis
- Heuristic ROI coverage simulation

Reinforcement learning is intentionally excluded from this phase.

## Run

From this folder:

```bash
python run_eda.py
python run_eda.py --splits val
python run_baseline.py --model yolo11s.pt --split val --sample-limit 50
python run_roi_analysis.py
```

`run_eda.py` uses `train val test` by default. Use `--splits val` for a quick smoke test.
For full validation baseline, omit `--sample-limit`.

## Baseline Model Choice

The baseline intentionally uses pretrained `yolo11s.pt` without VisDrone fine-tuning. This keeps full-image YOLO imperfect enough to study:

- small-object failures
- confidence behavior
- whether SAHI and adaptive ROI slicing are justified

Because pretrained YOLO uses COCO class IDs, strict VisDrone class-ID matching is not a perfect dataset-specific mAP. Treat this baseline primarily as failure-analysis evidence, not as a final VisDrone leaderboard score.

```bash
python run_baseline.py --model yolo11s.pt --split val
python run_baseline.py --model yolo11s.pt --split val --class-agnostic
```

`run_train_yolo.py` exists only as an optional future utility and is not part of the current ROI-SAHI baseline workflow.

## Outputs

- `outputs/dataset_stats.json`
- `outputs/dataset_validation.json`
- `outputs/eda_statistics.json`
- `outputs/baseline_metrics.json`
- `outputs/roi_metrics.json`
- `eda/*.png`
- `baseline/*.png`
- `roi_analysis/*.png`
