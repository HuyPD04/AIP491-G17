import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.visdrone import load_yolo_labels

def visualize_yolo_label_on_image(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh tại đường dẫn '{image_path}'")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_h, img_w, _ = img.shape

    objects = load_yolo_labels(label_path)
    if not objects:
        print(f"Lỗi: Không thể tải nhãn tại đường dẫn '{label_path}'")
        return

    obj = objects[70]
    
    x_center, y_center, bbox_w, bbox_h = obj.bbox
    
    xmin = int((x_center - bbox_w / 2) * img_w)
    ymin = int((y_center - bbox_h / 2) * img_h)
    xmax = int((x_center + bbox_w / 2) * img_w)
    ymax = int((y_center + bbox_h / 2) * img_h)

    color = (255, 0, 0) 
    thickness = 1

    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    img_with_bbox = cv2.rectangle(img_rgb, start_point, end_point, color, thickness)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_bbox)
    plt.axis('off') 
    plt.title(f"{obj.class_id} - {obj.class_name}")
    plt.show()

if __name__ == "__main__":
    sample_image_path = r"C:\Users\Admin\OneDrive\Desktop\DOAN\backup\data\processed\images\train\0000002_00005_d_0000014.jpg"
    sample_label_path = r"C:\Users\Admin\OneDrive\Desktop\DOAN\backup\data\processed\labels\train\0000002_00005_d_0000014.txt"

    visualize_yolo_label_on_image(sample_image_path, sample_label_path)