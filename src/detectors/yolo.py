from __future__ import annotations

import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

class YOLODetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        img_size: int = 640,
    ):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size

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
        pass

