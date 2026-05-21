from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from src.rl.action import history_shape
from src.utils.bbox import as_array, boxes_area, roi_to_state, is_small_object

# xây mảng thông tin cho biết lượt yolo đầu tiên phát hiện được bao nhiêu box, score trung bình, thấp nhất, cao nhất, ...
def build_detection_summary(predictions: dict | None, image_size: tuple[int, int], config: dict | None = None) -> np.ndarray:
    config = config or {}
    if not predictions:
        return np.zeros(6, dtype=np.float32)
    boxes = as_array(predictions.get("boxes")).reshape(-1, 4)
    scores = as_array(predictions.get("scores")).reshape(-1)
    if boxes.size == 0 or scores.size == 0:
        return np.zeros(6, dtype=np.float32)
    width, height = image_size
    image_area = max(float(width * height), 1.0)
    areas = boxes_area(boxes) / image_area
    low_conf_threshold = float(config.get("low_conf_threshold", 0.4))
    small_area_threshold = float(config.get("small_area_threshold", 0.01))
    return np.asarray(
        [
            float(len(boxes)),
            float(np.mean(scores)),
            float(np.min(scores)),
            float(np.sum(scores < low_conf_threshold)),
            float(np.mean(areas)) if len(areas) else 0.0,
            float(np.sum(areas < small_area_threshold)),
        ],
        dtype=np.float32,
    )

#xây heatmap 
def build_detection_map(predictions: dict | None, image_size: tuple[int, int], map_size: int, config: dict | None = None) -> np.ndarray:
    config = config or {}
    channels = _detection_map_channels(config)
    maps = {channel: np.zeros((map_size, map_size), dtype=np.float32) for channel in channels}
    if not predictions:
        return _stack_detection_maps(maps, channels)
    boxes = as_array(predictions.get("boxes")).reshape(-1, 4)
    scores = as_array(predictions.get("scores")).reshape(-1)
    if boxes.size == 0:
        return _stack_detection_maps(maps, channels)
    width, height = image_size
    low_conf_threshold = float(config.get("low_conf_threshold", 0.4))
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(float, box)
        col1 = int(np.floor((max(0.0, x1) / width) * map_size))
        col2 = int(np.ceil((min(float(width), x2) / width) * map_size))
        row1 = int(np.floor((max(0.0, y1) / height) * map_size))
        row2 = int(np.ceil((min(float(height), y2) / height) * map_size))
        col1 = max(0, min(map_size - 1, col1))
        row1 = max(0, min(map_size - 1, row1))
        col2 = max(col1 + 1, min(map_size, col2))
        row2 = max(row1 + 1, min(map_size, row2))
        score = float(score)
        if "score" in maps:
            maps["score"][row1:row2, col1:col2] = np.maximum(maps["score"][row1:row2, col1:col2], score)
        if "low_conf" in maps and score < low_conf_threshold:
            maps["low_conf"][row1:row2, col1:col2] = np.maximum(
                maps["low_conf"][row1:row2, col1:col2],
                1.0 - score,
            )
        if "small" in maps and is_small_object(box, image_size, config):
            maps["small"][row1:row2, col1:col2] = np.maximum(maps["small"][row1:row2, col1:col2], score)
    return _stack_detection_maps(maps, channels)

def build_state(
    current_roi: Sequence[float],
    history_map,
    predictions: dict | None,
    step_idx: int,
    config: dict,
    image_size: tuple[int, int],
):
    feature_dim = int(config.get("feature_dim", 0)) 
    feature_arr = np.zeros(feature_dim, dtype=np.float32)
    roi_arr = roi_to_state(current_roi, image_size)
    history_arr = as_array(history_map).reshape(-1).astype(np.float32)
    summary_arr = build_detection_summary(predictions, image_size, config)
    detection_map_arr = np.empty((0,), dtype=np.float32)
    if config.get("use_detection_map", False):
        map_size = history_shape(config)[0]
        detection_map_arr = build_detection_map(predictions, image_size, map_size, config).reshape(-1).astype(np.float32)
    max_steps = max(float(config.get("T_max", config.get("max_steps", 1))), 1.0)
    step_arr = np.asarray([float(step_idx) / max_steps], dtype=np.float32)
    state = np.concatenate([feature_arr, roi_arr, history_arr, detection_map_arr, summary_arr, step_arr]).astype(np.float32)
    
    return torch.from_numpy(state)

def state_dim(config: dict) -> int:
    grid_h, grid_w = history_shape(config)
    feature_dim = int(config.get("feature_dim", 0))
    detection_map_dim = grid_h * grid_w * len(_detection_map_channels(config)) if config.get("use_detection_map", False) else 0
    return feature_dim + 7 + grid_h * grid_w + detection_map_dim + 6 + 1

def _detection_map_channels(config: dict) -> list[str]:
    channels = config.get("detection_map_channels")
    if channels is None:
        return ["score"]
    if isinstance(channels, str):
        channels = [part.strip() for part in channels.split(",") if part.strip()]
    normalized = []
    for channel in channels:
        channel = str(channel)
        if channel not in {"score", "low_conf", "small"}:
            raise ValueError(f"Unknown detection map channel: {channel}")
        normalized.append(channel)
    return normalized or ["score"]


def _stack_detection_maps(maps: dict[str, np.ndarray], channels: list[str]) -> np.ndarray:
    if len(channels) == 1:
        return maps[channels[0]]
    return np.stack([maps[channel] for channel in channels], axis=0)
