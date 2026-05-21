from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


Box = Sequence[float]


def as_array(values, dtype=np.float32) -> np.ndarray:
    if values is None:
        return np.empty((0,), dtype=dtype)
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def xyxy_area(box: Box) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def boxes_area(boxes) -> np.ndarray:
    boxes = as_array(boxes).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: Box, box_b: Box) -> float:
    intersection = intersection_area(box_a, box_b)
    union = xyxy_area(box_a) + xyxy_area(box_b) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def pairwise_iou(boxes_a, boxes_b) -> np.ndarray:
    a = as_array(boxes_a).reshape(-1, 4)
    b = as_array(boxes_b).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = boxes_area(a)[:, None] + boxes_area(b)[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def containment_ratio(inner: Box, outer: Box) -> float:
    area = xyxy_area(inner)
    if area <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / area


def max_iou(roi: Box, boxes: Iterable[Box]) -> float:
    best = 0.0
    for box in boxes:
        best = max(best, iou(roi, box))
    return best


def box_center(box: Box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def center_inside(box: Box, roi: Box) -> bool:
    cx, cy = box_center(box)
    return float(roi[0]) <= cx <= float(roi[2]) and float(roi[1]) <= cy <= float(roi[3])


def clamp_roi(
    roi: Box,
    image_size: tuple[int | float, int | float],
    min_width: float = 1.0,
    min_height: float = 1.0,
) -> list[float]:
    width, height = map(float, image_size)
    x1, y1, x2, y2 = map(float, roi)
    roi_w = min(max(x2 - x1, min_width), width)
    roi_h = min(max(y2 - y1, min_height), height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    x1 = cx - roi_w / 2.0
    x2 = cx + roi_w / 2.0
    y1 = cy - roi_h / 2.0
    y2 = cy + roi_h / 2.0
    if x1 < 0.0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0.0:
        y2 -= y1
        y1 = 0.0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    return [max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)]


def is_valid_roi(roi: Box, image_size: tuple[int | float, int | float], config: dict | None = None) -> bool:
    config = config or {}
    width, height = map(float, image_size)
    x1, y1, x2, y2 = map(float, roi)
    min_w = float(config.get("min_roi_width", config.get("min_crop_size", 1)))
    min_h = float(config.get("min_roi_height", config.get("min_crop_size", 1)))
    max_area_ratio = float(config.get("max_roi_area_ratio", 1.0))
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if (x2 - x1) < min_w or (y2 - y1) < min_h:
        return False
    return xyxy_area(roi) <= width * height * max_area_ratio + 1e-6


def roi_to_state(roi: Box, image_size: tuple[int | float, int | float]) -> np.ndarray:
    width, height = map(float, image_size)
    x1, y1, x2, y2 = map(float, roi)
    roi_w = max(0.0, x2 - x1)
    roi_h = max(0.0, y2 - y1)
    image_area = max(width * height, 1.0)
    return np.asarray(
        [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
            roi_w / width,
            roi_h / height,
            (roi_w * roi_h) / image_area,
        ],
        dtype=np.float32,
    )


def is_detected(
    gt_class_id: int,
    gt_box: tuple[float, float, float, float],
    predictions: list[tuple[int, tuple[float, float, float, float]]],
    CLASS_AWARE: bool = False,
    IOU_THRESHOLD: float = 0.5,
) -> bool:
    for pred_class_id, pred_box in predictions:
        if CLASS_AWARE and gt_class_id != pred_class_id:
            continue
        if iou(gt_box, pred_box) >= IOU_THRESHOLD:
            return True
    return False

def is_small_object(box, image_size: tuple[int, int], config: dict) -> bool:
    width, height = image_size
    area = float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    return area / max(float(width * height), 1.0) < float(config.get("small_area_threshold", 0.01))