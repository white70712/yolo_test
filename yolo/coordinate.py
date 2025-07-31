import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 初始化 YOLOv8 模型
model = YOLO("yolov10m.pt")  # 可改 yolov8n.pt 等

# RealSense 設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 嘗試啟動攝影機
try:
    profile = pipeline.start(config)
    device = profile.get_device()
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"✅ 已啟用裝置：{name}，序號：{serial}")
except Exception as e:
    print("❌ RealSense 啟動失敗：", e)
    exit(1)

# 對齊設定
align = rs.align(rs.stream.color)

# 取得深度比例尺
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"📏 深度比例尺：{depth_scale} 米/單位")

print("🚀 開始偵測（按 Q 鍵結束）")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("⚠️ 無法取得影像")
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # YOLO 偵測
        results = model(color_image)[0]
        annotated_image = results.plot()

        # 處理每個偵測框
        for box in results.boxes:
            # 邊框與標籤
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 深度值
            depth = depth_frame.get_distance(cx, cy)

            # 空間座標
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

            # 類別名稱
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # 顯示標籤
            label = f"{class_name}: ({X:.2f}, {Y:.2f}, {Z:.2f}) m"
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 顯示畫面
        cv2.imshow("YOLOv8 + RealSense 3D", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🛑 已結束偵測")
