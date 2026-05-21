from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.visdrone import load_yolo_label_file
from src.rl.action import Action, coerce_action, history_shape, initial_roi, update_history_map, update_roi
from src.rl.reward import compute_reward
from src.rl.state import build_state
from src.utils.bbox import is_valid_roi


class AdaptiveSlicingEnv:
    def __init__(self, config: dict, dataset: Sequence[dict]):
        if not dataset:
            raise ValueError("AdaptiveSlicingEnv requires a non-empty dataset")
        self.config = dict(config)
        self.dataset = list(dataset)
        self._next_index = 0
        self.sample: dict | None = None
        self.image_size = (640, 640)
        self.current_roi: list[float] = [0.0, 0.0, 1.0, 1.0]
        self.history_map = np.zeros(history_shape(self.config), dtype=np.float32)
        self.visited_rois: list[list[float]] = []
        self.covered_object_ids: set = set()
        self.hard_objects: list[dict] = []
        self.full_predictions: dict = {"boxes": [], "scores": [], "classes": []}
        self.step_idx = 0
        self.cumulative_reward = 0.0
        self.delayed_positive_reward = 0.0

    def reset(self, image_id: str | None = None):
        self.sample = self._select_sample(image_id)
        width = int(self.sample.get("width", 1))
        height = int(self.sample.get("height", 1))
        self.image_size = (width, height)
        self.config["image_size"] = self.image_size
        self.history_map = np.zeros(history_shape(self.config), dtype=np.float32)
        self.visited_rois = []
        self.covered_object_ids = set()
        self.hard_objects = self._load_hard_regions(self.sample.get("hard_region_path"))
        self.full_predictions = self._load_predictions(self.sample.get("prediction_path"))
        self.config["full_predictions"] = self.full_predictions
        self.current_roi = initial_roi(self.image_size, self.config)
        self.step_idx = 0
        self.cumulative_reward = 0.0
        self.delayed_positive_reward = 0.0
        return self._state()

    def step(self, action: Action | int):
        action = coerce_action(action)
        if action is Action.STOP:
            reward = -float(self.config.get("lambda_step", 0.01))
            if bool(self.config.get("terminal_reward_only", False)):
                reward += self.delayed_positive_reward
                self.delayed_positive_reward = 0.0
            self.cumulative_reward += reward
            return self._state(), reward, True, {"action": action.name, "stopped": True}

        next_roi = update_roi(self.current_roi, action, self.image_size, self.config)
        reward, info = compute_reward(
            next_roi,
            self.hard_objects,
            self.visited_rois,
            self.covered_object_ids,
            self.image_size,
            self.config,
        )
        self.current_roi = next_roi
        if is_valid_roi(next_roi, self.image_size, self.config):
            self.visited_rois.append(list(next_roi))
            self.history_map = update_history_map(self.history_map, next_roi, self.image_size)

        self.step_idx += 1
        done = self.step_idx >= int(self.config.get("T_max", self.config.get("max_steps", 7)))
        max_slices = self.config.get("max_slices", self.config.get("num_slices"))
        if max_slices is not None:
            done = done or len(self.visited_rois) >= int(max_slices)
        if bool(self.config.get("terminal_reward_only", False)):
            positive_reward = float(info.get("positive_reward", 0.0))
            self.delayed_positive_reward += positive_reward
            reward -= positive_reward
            if done:
                reward += self.delayed_positive_reward
                self.delayed_positive_reward = 0.0
        self.cumulative_reward += reward
        info.update(
            {
                "action": action.name,
                "roi": list(next_roi),
                "num_slices": len(self.visited_rois),
                "done": done,
            }
        )
        return self._state(), reward, done, info

    def render(self) -> dict:
        return {
            "image_id": self.sample.get("image_id") if self.sample else None,
            "current_roi": list(self.current_roi),
            "visited_rois": list(self.visited_rois),
            "covered_object_ids": sorted(self.covered_object_ids),
            "step_idx": self.step_idx,
            "cumulative_reward": self.cumulative_reward,
        }

    def _select_sample(self, image_id: str | None) -> dict:
        if image_id is not None:
            for sample in self.dataset:
                if sample.get("image_id") == image_id:
                    return sample
            raise KeyError(f"image_id not found in dataset: {image_id}")
        sample = self.dataset[self._next_index % len(self.dataset)]
        self._next_index += 1
        return sample

    def _load_hard_regions(self, path_value) -> list[dict]:
        if not path_value:
            return []
        path = Path(path_value)
        if not path.exists():
            return []
        return load_yolo_label_file(path, self.image_size)

    def _load_predictions(self, path_value) -> dict:
        if not path_value:
            return {"boxes": [], "scores": [], "classes": []}
        path = Path(path_value)
        if not path.exists():
            return {"boxes": [], "scores": [], "classes": []}
        boxes = []
        scores = []
        classes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id_s, score_s, x1_s, y1_s, x2_s, y2_s = line.split()[:6]
            classes.append(int(float(class_id_s)))
            scores.append(float(score_s))
            boxes.append([float(x1_s), float(y1_s), float(x2_s), float(y2_s)])
        return {"boxes": boxes, "scores": scores, "classes": classes}

    def _state(self):
        return build_state(
            self.current_roi,
            self.history_map,
            self.full_predictions,
            self.step_idx,
            self.config,
            self.image_size,
        )
