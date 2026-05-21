from __future__ import annotations

import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

ImageSize = tuple[int, int]  


COCO_TO_VISDRONE = {
    0: 0,  # person -> pedestrian
    1: 2,  # bicycle
    2: 3,  # car
    3: 9,  # motorcycle -> motor
    5: 8,  # bus
    7: 5,  # truck
}

class YOLODetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        img_size: int = 640,
        class_map_mode: str | None = None,
    ):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        self.class_map_mode = class_map_mode

    def _predict(self, image):
        results = self.model.predict(
            image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        if not results: 
            return self._empty_prediction()
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._empty_prediction()
        
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
        xyxy, scores, classes = self._map_classes(xyxy, scores, classes)
        return {
            "boxes": xyxy,
            "scores": scores,
            "classes": classes,
        }
        
    def _empty_prediction(self):
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "classes": np.empty((0,), dtype=np.int64),
        }
    
    def detect_full_image(self, image):
        return self._predict(image)
    
    def detect_crop(self, crop):
        return self._predict(crop)
    
    def extract_features(self, image):
        return np.empty((0,), dtype=np.float32)

    def _map_classes(self, boxes, scores, classes):
        if self.class_map_mode in (None, "native", "none"):
            return boxes, scores, classes
        if self.class_map_mode != "coco_to_visdrone":
            raise ValueError(f"Unknown class_map_mode: {self.class_map_mode}")
        mapped_boxes = []
        mapped_scores = []
        mapped_classes = []
        for box, score, class_id in zip(boxes, scores, classes):
            class_id = int(class_id)
            if class_id not in COCO_TO_VISDRONE:
                continue
            mapped_boxes.append(box)
            mapped_scores.append(score)
            mapped_classes.append(COCO_TO_VISDRONE[class_id])
        if not mapped_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        return (
            np.asarray(mapped_boxes, dtype=np.float32),
            np.asarray(mapped_scores, dtype=np.float32),
            np.asarray(mapped_classes, dtype=np.int64),
        )

def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image: ImageSize,
) -> tuple[float, float, float, float]:
    image_width, image_height = image
    xmin = (x_center - box_width / 2) * image_width
    ymin = (y_center - box_height / 2) * image_height
    xmax = (x_center + box_width / 2) * image_width
    ymax = (y_center + box_height / 2) * image_height
    return [xmin, ymin, xmax, ymax]

def load_ground_truth(
    label_path: Path,
    image_size: ImageSize,
) -> list[tuple[int, tuple[float, float, float, float], str]]:
    image_width, image_height = image_size
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id_str, x_str, y_str, w_str, h_str = line.split()
        class_id = int(class_id_str)
        x_center = float(x_str)
        y_center = float(y_str)
        box_width = float(w_str)
        box_height = float(h_str)
        xyxy = yolo_to_xyxy(
            x_center,
            y_center,
            box_width,
            box_height,
            image_size,
        )
        yolo_line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}\n"
        )
        boxes.append((class_id, xyxy, yolo_line))

    return boxes

def load_predictions(
    prediction_path: Path,
    MIN_CONFIDENCE: float = 0.25,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes = []

    if not prediction_path.exists():
        return boxes

    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id_str, score_str, xmin_str, ymin_str, xmax_str, ymax_str = line.split()
        score = float(score_str)
        if score < MIN_CONFIDENCE:
            continue

        class_id = int(float(class_id_str))
        xyxy = tuple(map(float, (xmin_str, ymin_str, xmax_str, ymax_str)))
        boxes.append((class_id, xyxy))

    return boxes