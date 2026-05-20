import cv2
import matplotlib.pyplot as plt

def visualize_pixel_bbox(image_path, xmin, ymin, xmax, ymax):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh tại đường dẫn '{image_path}'")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start_point = (int(xmin), int(ymin))
    end_point = (int(xmax), int(ymax))

    color = (255, 0, 0) 
    thickness = 2

    img_with_bbox = cv2.rectangle(img_rgb, start_point, end_point, color, thickness)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_bbox)
    plt.axis('off') 
    plt.title("Visualize Pixel Bounding Box (XYXY)")
    plt.show()

if __name__ == "__main__":
    sample_image_path = r"C:\Users\Admin\OneDrive\Desktop\DOAN\backup\data\processed\images\train\0000002_00005_d_0000014.jpg"
    
    xmin, ymin, xmax, ymax = 637.10, 422.13, 700.41, 471.89
    
    visualize_pixel_bbox(sample_image_path, xmin, ymin, xmax, ymax)