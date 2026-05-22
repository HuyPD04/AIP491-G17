# Data Research Report

- Splits: train, val, test
- Total images: 8629
- Total labels: 457066
- Total small labels: 285028
- Small label ratio: 0.6236

## Split Summary
- train: images=6471, labels=343205, small=207605, avg_labels/image=53.04
- val: images=548, labels=38759, small=26586, avg_labels/image=70.73
- test: images=1610, labels=75102, small=50837, avg_labels/image=46.65

## Difficulty Groups
- high_density: top image=0000059_01886_d_0000114 (902 labels, 902 small)
- many_small_objects: top image=0000059_01886_d_0000114 (902 labels, 902 small)
- high_small_ratio: top image=0000040_04284_d_0000071 (131 labels, 131 small)
- border_heavy: top image=0000059_01886_d_0000114 (902 labels, 902 small)
- low_density: top image=0000099_00149_d_0000001 (1 labels, 0 small)

## ROI Oracle
- Best-by-efficiency counts: {'center': 3373, 'density': 218, 'horizon': 4764, 'uncertain': 274}
- Best-by-coverage counts: {'center': 672, 'density': 3327, 'horizon': 2527, 'uncertain': 2103}

This report is data-only and does not use model predictions.