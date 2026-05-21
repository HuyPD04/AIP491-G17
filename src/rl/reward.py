from __future__ import annotations

from typing import MutableSet, Sequence

from src.utils.bbox import containment_ratio, intersection_area, is_valid_roi, max_iou, xyxy_area


def compute_reward(
    roi: Sequence[float],
    hard_objects: list[dict],
    visited_rois: list[Sequence[float]],
    covered_object_ids: MutableSet,
    image_size: tuple[int | float, int | float],
    config: dict,
) -> tuple[float, dict]:
    width, height = map(float, image_size)
    image_area = max(width * height, 1.0)
    area_ratio = xyxy_area(roi) / image_area
    valid = is_valid_roi(roi, image_size, config)

    positive_reward = 0.0
    num_new_objects = 0
    newly_covered = []
    if valid:
        for index, obj in enumerate(hard_objects):
            obj_id = obj.get("id", obj.get("object_id", index))
            if obj_id in covered_object_ids:
                continue
            box = obj.get("bbox_xyxy")
            if box is None:
                continue
            obj_area = max(xyxy_area(box), 1.0)
            containment_score = intersection_area(box, roi) / obj_area
            if containment_score < float(config.get("object_containment_threshold", config.get("min_coverage", 0.5))):
                continue
            hard_score = float(obj.get("hard_score", 1.0))
            object_reward = hard_score * containment_score
            if config.get("scale_positive_by_area", False):
                area_power = float(config.get("positive_area_power", 1.0))
                min_scale = float(config.get("min_positive_area_scale", 0.05))
                area_scale = max(min_scale, (1.0 - area_ratio) ** area_power)
                object_reward *= area_scale
            positive_reward += object_reward
            num_new_objects += 1
            newly_covered.append(obj_id)

    for obj_id in newly_covered:
        covered_object_ids.add(obj_id)

    lambda_area = float(config.get("lambda_area", config.get("area_penalty", 0.1)))
    area_penalty = lambda_area * area_ratio
    duplicate_iou = max_iou(roi, visited_rois) if visited_rois else 0.0
    duplicate_penalty = 0.0
    duplicate_threshold = float(config.get("duplicate_iou_threshold", config.get("max_slice_overlap", 0.8)))
    if duplicate_iou > duplicate_threshold:
        duplicate_penalty = float(config.get("lambda_duplicate", config.get("repeat_penalty", 0.3)))

    full_pred_overlap = _max_full_prediction_overlap(roi, config)
    full_pred_roi_coverage = _full_prediction_roi_coverage(roi, config)
    full_pred_overlap_penalty = 0.0
    if full_pred_roi_coverage > float(config.get("full_pred_roi_coverage_threshold", 0.15)):
        full_pred_overlap_penalty = float(config.get("lambda_full_pred_overlap", 0.25)) * full_pred_roi_coverage

    step_penalty = float(config.get("lambda_step", 0.01))
    invalid_penalty = 0.0 if valid else float(config.get("lambda_invalid", config.get("invalid_crop_penalty", 1.0)))
    empty_penalty = 0.0
    if valid and num_new_objects == 0 and config.get("use_empty_crop_penalty", False):
        empty_penalty = float(config.get("empty_crop_penalty", 0.2))

    reward = positive_reward - area_penalty - duplicate_penalty - full_pred_overlap_penalty - step_penalty - invalid_penalty - empty_penalty
    info = {
        "positive_reward": float(positive_reward),
        "area_penalty": float(area_penalty),
        "duplicate_penalty": float(duplicate_penalty),
        "full_pred_overlap_penalty": float(full_pred_overlap_penalty),
        "empty_penalty": float(empty_penalty),
        "step_penalty": float(step_penalty),
        "invalid_penalty": float(invalid_penalty),
        "num_new_objects": int(num_new_objects),
        "num_total_covered_objects": int(len(covered_object_ids)),
        "max_duplicate_iou": float(duplicate_iou),
        "max_full_pred_overlap": float(full_pred_overlap),
        "full_pred_roi_coverage": float(full_pred_roi_coverage),
        "object_containment_threshold": float(config.get("object_containment_threshold", config.get("min_coverage", 0.5))),
        "valid": bool(valid),
    }
    return float(reward), info


def _max_full_prediction_overlap(roi: Sequence[float], config: dict) -> float:
    predictions = config.get("full_predictions") or {}
    boxes = predictions.get("boxes") or []
    scores = predictions.get("scores") or []
    min_score = float(config.get("full_pred_overlap_min_score", 0.4))
    best = 0.0
    for box, score in zip(boxes, scores):
        if float(score) < min_score:
            continue
        best = max(best, containment_ratio(box, roi))
    return best


def _full_prediction_roi_coverage(roi: Sequence[float], config: dict) -> float:
    predictions = config.get("full_predictions") or {}
    boxes = predictions.get("boxes") or []
    scores = predictions.get("scores") or []
    min_score = float(config.get("full_pred_overlap_min_score", 0.4))
    roi_area = max(xyxy_area(roi), 1.0)
    covered_area = 0.0
    for box, score in zip(boxes, scores):
        if float(score) < min_score:
            continue
        covered_area += intersection_area(box, roi)
    return min(1.0, covered_area / roi_area)
