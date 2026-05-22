# Output Structure

## final

Core files for the current report and result discussion.

- `data_research_report.json`
- `data_research_report.md`
- `dataset_validation.json`
- `per_image_data_index.csv`
- `difficulty_groups.json`
- `baseline_metrics.json`
- `sahi_metrics.json`
- `test_test_analysis.json`
- `test_test_analysis_report.md`
- `roi_metrics.json`
- `sahi_test_predata_summary.json`
- `figures/`

## detailed

Detailed error-analysis files. Keep these when inspecting false positives, false negatives, latency, or per-image metadata.

- `baseline_false_negatives.json`
- `baseline_false_positives.json`
- `sahi_false_negatives.json`
- `sahi_false_positives.json`
- `baseline_latency_ms.json`
- `sahi_latency_ms.json`
- `per_image_data_index.json`

## heavy

Large intermediate files that are useful for reproducibility or later SAHI/adaptive ROI work, but are not needed for quick report reading.

- `baseline_predictions.json`
- `sahi_predictions.json`
- `sahi_test_predata.json`
- `roi_oracle_per_image.json`

## archive

Older or auxiliary outputs from earlier validation-only and EDA runs.

- `val_analysis/`
- `old_eda/`
- `optional_training/`
- `figures/`

