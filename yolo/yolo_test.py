from ultralytics import YOLO
import cv2

# 載入模型
model = YOLO("yolov8s.pt")
# 推論一張圖片（網路圖片會自動下載）
results = model("https://thumb.photo-ac.com/47/47846b74688ea02dc9e7d961d48165d4_t.jpeg")

# 顯示標註後圖片
annotated_frame = results[0].plot()

cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
