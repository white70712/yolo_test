import cv2
from ultralytics import YOLO

# 初始化 YOLO 模型
model = YOLO("yolov8n.pt")  # 或 yolov8s.pt

# 開啟 USB 攝影機（0 表預設設備）
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ 攝影機啟動失敗，請確認是否接好")
    exit()

print("✅ 攝影機啟動成功，按 Q 鍵離開")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 無法取得畫面")
        break

    # YOLO 偵測
    results = model(frame)[0]
    annotated_frame = results.plot()

    # 顯示畫面
    cv2.imshow("YOLOv8 - USB Camera", annotated_frame)

    # 按下 Q 結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 已結束偵測")
