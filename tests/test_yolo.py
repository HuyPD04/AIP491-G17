from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detectors.yolo import YOLODetector
from src.utils.config import load_config
from src.utils.setting import YOLO_CONFIG

def test_yolo_detector():
    config = load_config(YOLO_CONFIG)
    detector = YOLODetector(model_path=config["model_path"])

    image_path = Path(r"C:\Users\Admin\OneDrive\Desktop\DOAN\backup\data\processed\images\test\0000006_00159_d_0000001.jpg")
    results = detector.detect_full_image(image_path)
    
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    class_names = detector.model.names

    for box, score, class_id in zip(results["boxes"], results["scores"], results["classes"]):
        xmin, ymin, xmax, ymax = map(int, box)
        class_name = class_names.get(int(class_id), f"class_{class_id}")
        label = f"{class_name}: {score:.2f}"

        color = (0, 255, 0)
        cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), color, 2)
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_rgb, (xmin, ymin), (xmin + text_size[0], ymin - text_size[1] - 4), color, -1)
        cv2.putText(img_rgb, label, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_yolo_detector()