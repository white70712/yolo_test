import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
CONF_THRESHOLD = 0.4

# ------------------- YOLO 模型 -------------------
model = YOLO('yolo_wine.pt')  # 只使用自訂模型
names = model.names

# ------------------- RealSense 初始化 -------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# ✅ 取得內參
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# ✅ 像素 bbox 轉換為 3D 座標
def get_3d_point_from_bbox(bbox, depth_frame, intrinsics):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    depth = depth_frame.get_distance(cx, cy)
    if depth == 0:
        return None
    point3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
    return point3d  # [x, y, z] in meters
def get_ids_by_name(detections, target_name, top_n=1):
    """
    回傳 name 符合的物件 id（已依 z 軸由遠到近排序）
    """
    matched = [d for d in detections if d["name"] == target_name]
    if not matched:
        return None if top_n == 1 else []
    
    if top_n == 1:
        return matched[0]["id"]
    return [d["id"] for d in matched[:top_n]]

# ------------------- 主程式 -------------------
print("[INFO] 相機初始化完成，開始推論...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)[0]
        boxes = [b for b in results.boxes if b.conf[0].item() > CONF_THRESHOLD]

        # 儲存偵測結果
        detections = []

        # 先收集所有框 + 3D 座標
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = names.get(class_id, f"class_{class_id}")
            label = f"{name} {conf:.2f}"
            color = (0, 255, 0)  # 綠色
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            pos3d = get_3d_point_from_bbox([x1, y1, x2, y2], depth_frame, intrinsics)

            if pos3d:
                x3d, y3d, z3d = pos3d
                detections.append({
                    "name": name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "position_3d": {
                        "x": round(x3d, 4),
                        "y": round(y3d, 4),
                        "z": round(z3d, 4)
                    }
                })

        # 依 z 軸由遠到近排序
        detections.sort(key=lambda d: d["position_3d"]["z"], reverse=True)

        # 加入排序後 ID
        for i, det in enumerate(detections):
            det["id"] = i

        # ✅ 假設我今天想要找 "wine"
        
        cv2.imshow("YOLO + Depth", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/latest.json", "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=4, ensure_ascii=False)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
