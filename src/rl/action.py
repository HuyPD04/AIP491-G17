from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np

from src.utils.bbox import clamp_roi, iou


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    ZOOM_IN = 4
    ZOOM_OUT = 5
    STOP = 6


def coerce_action(action: Action | int) -> Action:
    if isinstance(action, Action):
        return action
    return Action(int(action))


def action_count() -> int:
    return len(Action)


def enabled_action_values(config: dict) -> list[int]:
    values = config.get("enabled_actions")
    if values is not None:
        return [int(value) for value in values]
    values = [
        Action.MOVE_UP.value,
        Action.MOVE_DOWN.value,
        Action.MOVE_LEFT.value,
        Action.MOVE_RIGHT.value,
    ]
    if bool(config.get("use_zoom", False)):
        values.extend([Action.ZOOM_IN.value, Action.ZOOM_OUT.value])
    values.append(Action.STOP.value)
    return values


def initial_roi(image_size: tuple[int | float, int | float], config: dict) -> list[float]:
    width, height = map(float, image_size)
    mode = config.get("initial_roi", "center_fixed")
    if config.get("roi_mode") == "fixed" or mode == "center_fixed":
        crop_w = min(float(config.get("fixed_roi_width", config.get("crop_size", width))), width)
        crop_h = min(float(config.get("fixed_roi_height", config.get("crop_size", height))), height)
        cx = width / 2.0
        cy = height / 2.0
        return clamp_roi(
            [cx - crop_w / 2.0, cy - crop_h / 2.0, cx + crop_w / 2.0, cy + crop_h / 2.0],
            image_size,
            min_width=min(crop_w, width),
            min_height=min(crop_h, height),
        )
    if mode == "center_half":
        return [width * 0.25, height * 0.25, width * 0.75, height * 0.75]
    return [0.0, 0.0, width, height]


def update_roi(
    roi: Sequence[float],
    action: Action | int,
    image_size: tuple[int | float, int | float],
    config: dict,
) -> list[float]:
    action = coerce_action(action)
    if action is Action.STOP:
        return list(map(float, roi))

    x1, y1, x2, y2 = map(float, roi)
    roi_w = x2 - x1
    roi_h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    move_ratio = float(config.get("move_ratio", 0.2))
    zoom_in_ratio = float(config.get("zoom_in_ratio", config.get("zoom_ratio", 0.8)))
    zoom_out_ratio = float(config.get("zoom_out_ratio", 1.0 / max(zoom_in_ratio, 1e-6)))
    fixed_mode = config.get("roi_mode") == "fixed"

    if action is Action.MOVE_UP:
        cy -= move_ratio * roi_h
    elif action is Action.MOVE_DOWN:
        cy += move_ratio * roi_h
    elif action is Action.MOVE_LEFT:
        cx -= move_ratio * roi_w
    elif action is Action.MOVE_RIGHT:
        cx += move_ratio * roi_w
    elif action is Action.ZOOM_IN and bool(config.get("use_zoom", False)) and not fixed_mode:
        roi_w *= zoom_in_ratio
        roi_h *= zoom_in_ratio
    elif action is Action.ZOOM_OUT and bool(config.get("use_zoom", False)) and not fixed_mode:
        roi_w *= zoom_out_ratio
        roi_h *= zoom_out_ratio

    return clamp_roi(
        [cx - roi_w / 2.0, cy - roi_h / 2.0, cx + roi_w / 2.0, cy + roi_h / 2.0],
        image_size,
        min_width=float(config.get("min_roi_width", config.get("min_crop_size", 1))),
        min_height=float(config.get("min_roi_height", config.get("min_crop_size", 1))),
    )


def max_duplicate_iou(roi: Sequence[float], visited_rois: list[Sequence[float]]) -> float:
    if not visited_rois:
        return 0.0
    return max(iou(roi, visited) for visited in visited_rois)


def history_shape(config: dict) -> tuple[int, int]:
    shape = config.get("history_shape", config.get("history_map_size", [16, 16]))
    if isinstance(shape, int):
        return int(shape), int(shape)
    return int(shape[0]), int(shape[1])


def update_history_map(
    history_map,
    roi: Sequence[float],
    image_size: tuple[int | float, int | float],
) -> np.ndarray:
    history = np.asarray(history_map, dtype=np.float32).copy()
    grid_h, grid_w = history.shape
    width, height = map(float, image_size)
    x1, y1, x2, y2 = map(float, roi)
    col1 = int(np.floor((x1 / width) * grid_w))
    col2 = int(np.ceil((x2 / width) * grid_w))
    row1 = int(np.floor((y1 / height) * grid_h))
    row2 = int(np.ceil((y2 / height) * grid_h))
    col1 = max(0, min(grid_w - 1, col1))
    row1 = max(0, min(grid_h - 1, row1))
    col2 = max(col1 + 1, min(grid_w, col2))
    row2 = max(row1 + 1, min(grid_h, row2))
    history[row1:row2, col1:col2] = 1.0
    return history
