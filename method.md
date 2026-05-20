1. Xác định môi trường RL 
Trạng thái (State - S):


Vector đặc trưng (Feature Vector): Vector trích xuất từ mạng xương sống (backbone) của YOLO để lấy bối cảnh toàn cục của hình ảnh.
Lịch sử hành động (h): Một bản đồ nhị phân (binary map) ghi lại vị trí các mảnh cắt đã thực hiện để tránh việc agent chọn trùng lặp một vùng nhiều lần.
Hành động (Action - A): 
Giai đoạn 1: 
Cố định kích thước mảnh cắt: 320x320 hoặc 480x480, …
Việc chọn mảnh cắt sẽ được thực hiện qua các action sau: di chuyển lên, xuống, sang trái, sang phải
Giai đoạn 2: Sau khi giai đoạn 1 ổn định thì bắt đầu thêm action zoom in/out nhằm điều chỉnh kích thước có thể vừa khít chứa các vật thể nhỏ
Hàm phần thưởng (Reward - R):


Phần thưởng tích dương: Trao thưởng nếu mảnh cắt chứa các vật thể nhỏ thực tế (Ground Truth) mà YOLO ở bước ảnh gốc đã bỏ lỡ hoặc có độ tin cậy thấp (1-ci).
Hình phạt dư thừa (Cost Penalty): Trừ điểm dựa trên diện tích vùng cắt (bB) để khuyến khích agent chọn mảnh cắt nhỏ nhất và ít nhất có thể mà vẫn tìm thấy vật thể. Việc kiểm tra xem các vật thể nhỏ mà YOLO ko bắt được có nằm trong mảnh cắt hay không dựa vào duyệt bbox nhãn của vật thể đó so với bbox mảnh cắt.
Note:  là hyperparameter. Ngoài ra, có thể thêm hàm thưởng phạt về sau nếu cần.
2. Quy trình Huấn luyện
Giai đoạn chuẩn bị: Chạy YOLO trên toàn bộ tập dữ liệu huấn luyện để xác định các vùng "khó" (nơi YOLO đoán sai hoặc bỏ sót vật thể nhỏ).
Huấn luyện cộng tác (Collaborative Training):
Bước A: Cố định YOLO, huấn luyện Agent RL (Policy Network) để học cách chọn các vùng cắt mang lại phần thưởng cao nhất (vùng có nhiều vật thể nhỏ bị bỏ sót).
Bước B: Tinh chỉnh (fine-tune) YOLO trên các mảnh cắt mà RL vừa chọn để mô hình làm quen với đặc trưng vật thể nhỏ ở độ phân giải cao.
Chiến lược thăm dò: Sử dụng -greedy để agent thử nghiệm các kích thước mảnh cắt khác nhau, kết hợp với Guided Exploration (dựa trên nhãn Ground Truth) để nhanh chóng hội tụ vào các vùng có vật thể.
Note: Bước B chỉ thực hiện nếu chất lượng của YOLO trên vật thể quá kém.
3. Quy trình Suy luận 
Bước 1 - Full Inference (FI): YOLO thực hiện phát hiện trên toàn bộ ảnh ở độ phân giải thấp để tìm các vật thể lớn và tạo ra bản đồ trạng thái ban đầu (S0).
Bước 2 - Quyết định thích nghi: RL phân tích bản đồ trạng thái và đưa ra một chuỗi T hành động để chọn ra các vị trí và kích thước mảnh cắt tối ưu.
Bước 3 - Sliced Inference: Chỉ đưa các vùng ảnh đã được RL chọn vào YOLO để dự đoán lại ở độ phân giải gốc của mảnh cắt.
Bước 4 - Hợp nhất kết quả: Sử dụng Cluster-DIoU-NMS (CDN) hoặc NMS để gộp các hộp dự đoán từ bước FI và bước Sliced Inference. CDN giúp loại bỏ các hộp trùng lặp nhanh hơn bằng tính toán ma trận song song và chính xác hơn nhờ tính đến khoảng cách tâm các hộp.

