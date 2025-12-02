import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.7 # 新增：用於跨類別 NMS 的重疊度閾值

# ------------------- YOLO 模型 -------------------
model = YOLO('model/alltype_v9.pt') # 只使用自訂模型
names = model.names

# ✅ 設定要隱藏邊框的名稱列表
HIDE_NAMES = ["XARM", "Plate"]

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
    return point3d # [x, y, z] in meters

def get_ids_by_name(detections, target_name, top_n=1):
    """
    回傳 name 符合的物件 id（已依 z 軸由遠到近排序）
    """
    matched = [d for d in detections if d["name"] == target_name]
    if not matched:
        return None if top_n == 1 else []
    
    # 確保 matched 已經依 z 軸排序 (在主迴圈中已排序)
    if top_n == 1:
        return matched[0]["id"]
    return [d["id"] for d in matched[:top_n]]

# ------------------- 🚀 跨類別 NMS 輔助函式 (移到迴圈外) -------------------
def calculate_iou(boxA, boxB):
    """計算兩個 BBox 的 Intersection over Union (IoU)"""
    # 決定交集區域的座標
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 計算交集面積
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 計算兩個 BBox 的總面積
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 計算 IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
# --------------------------------------------------------------------------

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
        # 篩選出所有信心值高於閾值的框
        all_boxes = [b for b in results.boxes if b.conf[0].item() > CONF_THRESHOLD]

        # ------------------- 🚀 關鍵客製化 NMS 區塊 -------------------
        
        # 收集所有候選框的資料結構 (包含 bbox, conf, name)
        candidates = []
        for box in all_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            name = names.get(class_id, f"class_{class_id}")
            
            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "name": name,
            })

        # 根據信心值由高到低排序 (信心高的優先保留)
        candidates.sort(key=lambda c: c["conf"], reverse=True)

        final_candidates = []
        
        for cand_i in candidates:
            is_redundant = False
            for cand_j in final_candidates:
                # 計算當前候選框與已保留框的 IoU
                iou = calculate_iou(cand_i["bbox"], cand_j["bbox"])
                
                # 如果重疊度超過閾值，則視為同一個物件，由於 cand_i 的信心值較低 (或與 cand_j 相同)，
                # 則將其標記為多餘 (is_redundant)，不加入 final_candidates。
                if iou > IOU_THRESHOLD:
                    is_redundant = True
                    break
            
            if not is_redundant:
                final_candidates.append(cand_i)
                
        # ------------------- 🚀 關鍵客製化 NMS 區塊結束 -------------------
        
        # 儲存偵測結果
        detections = []

        # 迭代最終篩選出的候選框 (final_candidates)
        for cand in final_candidates:
            x1, y1, x2, y2 = cand["bbox"]
            conf = cand["conf"]
            name = cand["name"]
            
            # --- 繪製區塊：只繪製最終保留的框 ---
            if name not in HIDE_NAMES:
                label = f"{name} {conf:.2f}"
                color = (0, 255, 0) # 綠色
                # 繪製邊框和標籤
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
            # -----------------------------------------------

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
