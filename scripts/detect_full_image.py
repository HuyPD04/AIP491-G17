from __future__ import annotations
from pathlib import Path
import sys
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detectors.yolo import YOLODetector
from src.utils.setting import YOLO_CONFIG, DETECT_FULL_IMAGE_DIR, TRAIN_IMAGES_DIR
from src.utils.config import load_config

def main():
    cfg = load_config(YOLO_CONFIG)
    detector = YOLODetector(
        model_path=cfg["model_path"],
        conf_threshold=cfg.get("conf_threshold", 0.25),
        iou_threshold=cfg.get("iou_threshold", 0.5),
        device=cfg.get("device", "cpu"),
        img_size=cfg.get("img_size", 640),
    )

    DETECT_FULL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(TRAIN_IMAGES_DIR.glob("*.jpg"), desc="Detecting full images"):
        results = detector.detect_full_image(image_path)
        output_path = DETECT_FULL_IMAGE_DIR / image_path.name.replace(".jpg", ".txt")
        with open(output_path, "w") as f:
            for box, score, class_id in zip(results["boxes"], results["scores"], results["classes"]):
                xmin, ymin, xmax, ymax = box
                f.write(f"{class_id} {score:.4f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n")

if __name__ == "__main__":
    main()

