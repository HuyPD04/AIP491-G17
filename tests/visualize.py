import cv2
import matplotlib.pyplot as plt

def visualize_bbox(image_path, bbox_left, bbox_top, bbox_width, bbox_height):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh tại đường dẫn '{image_path}'")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start_point = (int(bbox_left), int(bbox_top))
    end_point = (int(bbox_left + bbox_width), int(bbox_top + bbox_height))

    color = (255, 0, 0) 
    thickness = 2

    img_with_bbox = cv2.rectangle(img_rgb, start_point, end_point, color, thickness)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_bbox)
    plt.axis('off') 
    plt.title("Visualize Bounding Box")
    plt.show()

if __name__ == "__main__":
    sample_image_path = r"C:\Users\Admin\OneDrive\Desktop\DOAN\backup\data\processed\images\test\0000006_00159_d_0000001.jpg" 
    
    b_left, b_top, b_width, b_height = 685,463,110,65
    
    visualize_bbox(sample_image_path, b_left, b_top, b_width, b_height)