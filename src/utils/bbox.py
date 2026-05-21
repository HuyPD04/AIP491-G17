from __future__ import annotations
import numpy as np

def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0
    return intersection / union

def is_detected(
    gt_class_id: int,
    gt_box: tuple[float, float, float, float],
    predictions: list[tuple[int, tuple[float, float, float, float]]],
    CLASS_AWARE: bool = False,
    IOU_THRESHOLD: float = 0.5
) -> bool:
    for pred_class_id, pred_box in predictions:
        if CLASS_AWARE and gt_class_id != pred_class_id:
            continue
        if iou(gt_box, pred_box) >= IOU_THRESHOLD:
            return True
    return False