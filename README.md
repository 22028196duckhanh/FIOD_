# FIOD
### vấn đề:
#### hàm loss consistency (của segmentation sẽ là KL divergence giữa 2 vector xác suất của 2 ảnh trên từng pixel) => cần xem chuyển sang model OD thì như nào (model YOLOv9 đưa ra output gồm những gì, ...)
	temp solution: 
		+
		+
#### làm sao để tính ra được các loss liên quan OD (bbox, cls, dfl) với dataset như vậy (vì bình thường train với file .yaml, giờ mình lại có kiểu data_loader) => tính được loss thì cập nhật trọng số thế nào (đang muốn chỉ tương tác với backbone vì việc trích xuất feature đang lấy từ backbone)
	temp solution: 
		+ tạm thời đọc yolov9 xem tính loss như nào, cách train của họ (cảm giác hơi lú do code họ siêu dài)
		+ đi tìm project tinh chỉnh yolov9 hoặc kiểu dựa dựa ý tưởng viết lại chiến lược train
		+
