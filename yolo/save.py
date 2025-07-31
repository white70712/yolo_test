import pyrealsense2 as rs
import numpy as np
import cv2
import json
from ultralytics import YOLO

target_classes = ["Gin", "JOHNNIE WALKER Whiskey", "JOHN BARR Whiskey", "Rum", "Tequila", "Wine column"]

# 載入兩個模型
model_bast = YOLO("nnew5.pt")
model_yolo = YOLO("yolov8m.pt")

# RealSense 初始化與設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    device = profile.get_device()
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"已啟用裝置：{name}，序號：{serial}")
except Exception as e:
    print("RealSense 啟動失敗，錯誤如下：")
    print(e)
    exit(1)

align = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"深度比例尺：{depth_scale} 米/單位")

print("開始偵測（按 Q 鍵結束）")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

# 用來儲存連續出現的物件
seen_objects = {}

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("無法取得影像，請檢查攝影機與 USB 連接")
            continue

        color_image = np.asanyarray(color_frame.get_data())

        results_bast = model_bast(color_image)[0]
        results_yolo = model_yolo(color_image)[0]

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics  # 取得相機內參

        bast_detections = []
        for box in results_bast.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model_bast.names[cls_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            distance = depth_frame.get_distance(cx, cy)
            confidence = float(box.conf[0])
            # 3D座標反投影（公尺）
            position = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance)
            bast_detections.append([x1, y1, x2, y2, cls_name, distance, confidence, position])

        yolo_detections = []
        for box in results_yolo.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model_yolo.names[cls_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            distance = depth_frame.get_distance(cx, cy)
            confidence = float(box.conf[0])
            # 3D座標反投影（公尺）
            position = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance)
            yolo_detections.append([x1, y1, x2, y2, cls_name, distance, confidence, position])

        filtered_yolo = []
        for ydet in yolo_detections:
            y_box = ydet[:4]
            discard = False
            for bdet in bast_detections:
                b_box = bdet[:4]
                if iou(y_box, b_box) > 0.5 and ydet[4] != bdet[4]:
                    discard = True
                    break
            if not discard:
                filtered_yolo.append(ydet)

        combined_img = color_image.copy()

        for det in bast_detections:
            x1, y1, x2, y2, cls_name, distance, conf, _ = det
            label = f"{cls_name} {distance:.2f}m"
            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(combined_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for det in filtered_yolo:
            x1, y1, x2, y2, cls_name, distance, conf, _ = det
            label = f"{cls_name} {distance:.2f}m"
            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(combined_img, label, (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 只保留指定類別，並統計出現次數
        for det in bast_detections + filtered_yolo:
            x1, y1, x2, y2, cls_name, distance, conf, position = det

            if cls_name not in target_classes:
                continue  # 忽略非指定類別

            key = (cls_name, x1, y1, x2, y2)
            if key not in seen_objects:
                seen_objects[key] = {
                    "count": 1,
                    "distance": distance,
                    "confidence": conf,
                    "position": position
                }
            else:
                seen_objects[key]["count"] += 1

        cv2.imshow("Bast + YOLOv8m + RealSense", combined_img)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            print("偵測手動停止中...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("已結束偵測")

    # 過濾：出現次數超過一定次數才寫入 JSON，且同類別只保留一次
    final_detections = []
    appeared_classes = set()
    for (cls_name, x1, y1, x2, y2), data in seen_objects.items():
        if data["count"] >= 3:
            if cls_name not in appeared_classes:
                final_detections.append({
                    "name": cls_name,
                    "bbox": [x1, y1, x2, y2],                     # 2D框座標 (像素)
                    "distance": round(data["distance"], 2),       # 深度距離 (公尺)
                    "confidence": round(data["confidence"], 2),
                    "position": [round(coord, 3) for coord in data["position"]]  # 3D座標 (公尺)
                })
                appeared_classes.add(cls_name)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(final_detections, f, indent=2, ensure_ascii=False)
    print("已輸出 JSON：output.json")
