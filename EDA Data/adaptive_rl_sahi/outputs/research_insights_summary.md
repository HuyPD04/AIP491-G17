# Research Insights Summary

- Dataset contains 548 images and 38759 labeled objects.
- Average density is 70.73 objects per image.
- Small objects represent 68.59% of all annotations using COCO thresholds.
- 11.51% of objects are within 32 px of an image border.
- Use spatial_heatmap.png to identify where small-object slicing should be concentrated.
- Use density_analysis.png to identify images where full-frame YOLO is likely to be stressed by clutter.

RL is intentionally excluded in this phase.