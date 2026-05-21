from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.setting import (  
    DETECT_FULL_IMAGE_DIR,
    HARD_REGION_DIR,
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    YOLO_CONFIG
)
from src.data.visdrone import load_ground_truth, load_predictions
from src.utils.bbox import iou, is_detected
from src.utils.config import load_config

def main():
    cfg = load_config(YOLO_CONFIG)

    HARD_REGION_DIR.mkdir(parents=True, exist_ok=True)
    for old_output_path in HARD_REGION_DIR.glob("*.txt"):
        old_output_path.unlink()

    total_missed = 0
    for label_path in tqdm(sorted(TRAIN_LABELS_DIR.glob("*.txt")), desc="Dumping hard regions"):
        image_path = TRAIN_IMAGES_DIR / f"{label_path.stem}.jpg"
        prediction_path = DETECT_FULL_IMAGE_DIR / label_path.name
        output_path = HARD_REGION_DIR / label_path.name

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            image_size = image.size

        ground_truth = load_ground_truth(label_path, image_size)
        predictions = load_predictions(prediction_path)

        missed_lines = [
            yolo_line
            for gt_class_id, gt_box, yolo_line in ground_truth
            if not is_detected(gt_class_id, gt_box, predictions, CLASS_AWARE=cfg['CLASS_AWARE'], IOU_THRESHOLD=cfg['iou_threshold'])
        ]

        if missed_lines:
            output_path.write_text("".join(missed_lines), encoding="utf-8")
            total_missed += len(missed_lines)
    print(f"Total missed objects: {total_missed}")
if __name__ == "__main__":
    main()
