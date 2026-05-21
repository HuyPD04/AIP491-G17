from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.action import Action, enabled_action_values, history_shape, initial_roi, update_history_map, update_roi
from src.rl.policy import DQNPolicy
from src.rl.state import build_state, state_dim
from src.utils.bbox import as_array, iou, is_valid_roi, pairwise_iou
from src.utils.config import load_config
from src.utils.setting import CHECKPOINT_DIR, INFER_OUTPUT_DIR, RL_CONFIG, TEST_IMAGES_DIR, YOLO_CONFIG


def main() -> None:
    args = parse_args()
    image_path = args.image or first_image(TEST_IMAGES_DIR)
    pred_output = args.pred_output or (INFER_OUTPUT_DIR / f"{image_path.stem}_rl.txt")
    roi_output = args.roi_output or (INFER_OUTPUT_DIR / f"{image_path.stem}_rois.txt")
    predictions, rois = run_inference(image_path, args.checkpoint, pred_output, roi_output, args.config, args.yolo_config)
    print(f"predictions={pred_output}")
    print(f"rois={roi_output}")
    print(f"num_predictions={len(predictions['scores'])}")
    print(f"num_rois={len(rois)}")


def run_inference(
    image_path: Path,
    checkpoint_path: Path,
    pred_output: Path | None = None,
    roi_output: Path | None = None,
    rl_config_path: Path = RL_CONFIG,
    yolo_config_path: Path = YOLO_CONFIG,
) -> tuple[dict[str, np.ndarray], list[list[float]]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {checkpoint_path}. Run scripts/train_rl.py first.")

    from src.detectors.yolo import YOLODetector

    rl_cfg = load_config(rl_config_path)
    yolo_cfg = load_config(yolo_config_path)
    detector = YOLODetector(
        model_path=resolve_model_path(yolo_cfg["model_path"]),
        conf_threshold=float(yolo_cfg.get("conf_threshold", 0.25)),
        iou_threshold=float(yolo_cfg.get("iou_threshold", 0.5)),
        device=yolo_cfg.get("device", "cpu"),
        img_size=int(yolo_cfg.get("img_size", yolo_cfg.get("image_size", 640))),
        class_map_mode=yolo_cfg.get("class_map_mode", "native"),
    )

    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    image_size = image.size
    full_predictions = detector.detect_full_image(image)
    rois = select_rois(full_predictions, image_size, rl_cfg, checkpoint_path)

    crop_predictions = []
    for roi in rois:
        crop = image.crop(tuple(map(int, roi)))
        crop_pred = detector.detect_crop(crop)
        crop_predictions.append(map_crop_predictions(crop_pred, roi))

    merged = merge_predictions([full_predictions, *crop_predictions])
    predictions = nms_predictions(
        merged,
        iou_threshold=float(rl_cfg.get("nms_iou_threshold", yolo_cfg.get("iou_threshold", 0.5))),
        diou_threshold=float(rl_cfg.get("diou_threshold", 0.6)),
        containment_threshold=float(rl_cfg.get("containment_threshold", 0.85)),
    )

    if pred_output is not None:
        write_prediction_text(pred_output, predictions)
    if roi_output is not None:
        write_rois_text(roi_output, rois)
    return predictions, rois


def select_rois(
    full_predictions: dict,
    image_size: tuple[int, int],
    config: dict,
    checkpoint_path: Path,
) -> list[list[float]]:
    import torch

    config = dict(config)
    config["image_size"] = image_size
    device = torch.device(str(config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    policy = DQNPolicy(state_dim(config), 7, int(config.get("hidden_dim", 128))).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    current_roi = initial_roi(image_size, config)
    history_map = np.zeros(history_shape(config), dtype=np.float32)
    rois: list[list[float]] = []
    max_steps = int(config.get("max_steps", 7))
    max_rois = int(config.get("num_slices", 4))

    for step_idx in range(max_steps):
        state = build_state(current_roi, history_map, full_predictions, step_idx, config, image_size)
        with torch.no_grad():
            tensor_state = state.float() if hasattr(state, "float") else torch.as_tensor(state, dtype=torch.float32)
            if tensor_state.dim() == 1:
                tensor_state = tensor_state.unsqueeze(0)
            tensor_state = tensor_state.to(device)
            q_values = policy(tensor_state)
            allowed_values = enabled_action_values(config)
            mask = torch.full_like(q_values, float("-inf"))
            mask[:, allowed_values] = 0.0
            ranked_actions = torch.argsort(q_values + mask, dim=1, descending=True)[0].tolist()

        action_id = first_valid_action(ranked_actions, current_roi, rois, image_size, config)
        if action_id is None or action_id == Action.STOP.value:
            break

        next_roi = update_roi(current_roi, action_id, image_size, config)
        current_roi = next_roi
        if is_valid_roi(next_roi, image_size, config):
            rois.append(next_roi)
            history_map = update_history_map(history_map, next_roi, image_size)
        if len(rois) >= max_rois:
            break

    if config.get("return_all_rois", False):
        if not rois and is_valid_roi(current_roi, image_size, config):
            rois.append(current_roi)
        return rois
    return [current_roi] if is_valid_roi(current_roi, image_size, config) else []


def first_valid_action(
    ranked_actions: list[int],
    current_roi: Sequence[float],
    rois: list[list[float]],
    image_size: tuple[int, int],
    config: dict,
) -> int | None:
    duplicate_threshold = float(config.get("max_slice_overlap", 0.6))
    for action_id in ranked_actions:
        if action_id == Action.STOP.value:
            return action_id
        next_roi = update_roi(current_roi, action_id, image_size, config)
        if not is_valid_roi(next_roi, image_size, config):
            continue
        if iou(next_roi, current_roi) > 0.999:
            continue
        if any(iou(next_roi, existing) > duplicate_threshold for existing in rois):
            continue
        return int(action_id)
    return None


def map_crop_predictions(predictions: dict, roi: Sequence[float]) -> dict[str, np.ndarray]:
    x1, y1, _, _ = map(float, roi)
    boxes = as_array(predictions.get("boxes")).reshape(-1, 4).copy()
    if boxes.size:
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1
    return {
        "boxes": boxes,
        "scores": as_array(predictions.get("scores")).reshape(-1),
        "classes": as_array(predictions.get("classes"), dtype=np.int64).reshape(-1),
    }


def merge_predictions(prediction_sets: list[dict]) -> dict[str, np.ndarray]:
    boxes = [as_array(pred.get("boxes")).reshape(-1, 4) for pred in prediction_sets]
    scores = [as_array(pred.get("scores")).reshape(-1) for pred in prediction_sets]
    classes = [as_array(pred.get("classes"), dtype=np.int64).reshape(-1) for pred in prediction_sets]
    return {
        "boxes": np.concatenate(boxes, axis=0) if boxes else np.empty((0, 4), dtype=np.float32),
        "scores": np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32),
        "classes": np.concatenate(classes, axis=0) if classes else np.empty((0,), dtype=np.int64),
    }


def nms_predictions(
    predictions: dict,
    iou_threshold: float = 0.5,
    diou_threshold: float = 0.6,
    containment_threshold: float = 0.85,
) -> dict[str, np.ndarray]:
    boxes = as_array(predictions.get("boxes")).reshape(-1, 4)
    scores = as_array(predictions.get("scores")).reshape(-1)
    classes = as_array(predictions.get("classes"), dtype=np.int64).reshape(-1)
    if len(scores) == 0:
        return {"boxes": boxes, "scores": scores, "classes": classes}

    keep: list[int] = []
    for class_id in sorted(set(classes.tolist())):
        idxs = np.where(classes == class_id)[0]
        order = idxs[np.argsort(scores[idxs])[::-1]]
        while len(order) > 0:
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            rest = order[1:]
            ious = pairwise_iou(boxes[[current]], boxes[rest])[0]
            dious = diou_values(boxes[current], boxes[rest], ious)
            containments = pairwise_max_containment(boxes[current], boxes[rest])
            order = rest[(ious <= iou_threshold) & (dious <= diou_threshold) & (containments <= containment_threshold)]
    keep = sorted(keep, key=lambda idx: float(scores[idx]), reverse=True)
    return {"boxes": boxes[keep], "scores": scores[keep], "classes": classes[keep]}


def pairwise_max_containment(box, boxes) -> np.ndarray:
    boxes = as_array(boxes).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    box = as_array(box).reshape(4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-6)
    boxes_area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    return np.maximum(inter / box_area, inter / boxes_area).astype(np.float32)


def diou_values(box, boxes, ious) -> np.ndarray:
    cx1 = (box[0] + box[2]) / 2.0
    cy1 = (box[1] + box[3]) / 2.0
    cx2 = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy2 = (boxes[:, 1] + boxes[:, 3]) / 2.0
    center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    enc_x1 = np.minimum(box[0], boxes[:, 0])
    enc_y1 = np.minimum(box[1], boxes[:, 1])
    enc_x2 = np.maximum(box[2], boxes[:, 2])
    enc_y2 = np.maximum(box[3], boxes[:, 3])
    diag = np.maximum((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2, 1e-6)
    return ious - center_dist / diag


def write_prediction_text(path: Path, predictions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes = as_array(predictions.get("boxes")).reshape(-1, 4)
    scores = as_array(predictions.get("scores")).reshape(-1)
    classes = as_array(predictions.get("classes"), dtype=np.int64).reshape(-1)
    lines = []
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        lines.append(f"{int(class_id)} {float(score):.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_rois_text(path: Path, rois: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}" for x1, y1, x2, y2 in rois]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def first_image(image_dir: Path) -> Path:
    for image_path in sorted(image_dir.glob("*.jpg")):
        return image_path
    raise FileNotFoundError(f"No .jpg image found in {image_dir}")


def resolve_model_path(path_value: str | Path) -> str:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return str(path)
    backup_path = ROOT / path
    return str(backup_path if backup_path.exists() else path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL adaptive slicing inference on one image.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "final.pt")
    parser.add_argument("--pred-output", type=Path)
    parser.add_argument("--roi-output", type=Path)
    parser.add_argument("--config", type=Path, default=RL_CONFIG)
    parser.add_argument("--yolo-config", type=Path, default=YOLO_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    main()
