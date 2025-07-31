import pyrealsense2 as rs
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from datetime import datetime

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.4

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * (yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def get_distance(depth_frame, x, y):
    return round(depth_frame.get_distance(x, y), 3)

def get_position(depth_frame, x, y, depth_intrin):
    distance = depth_frame.get_distance(x, y)
    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], distance)
    return {'x': round(point[0], 3), 'y': round(point[1], 3), 'z': round(point[2], 3)}

model_base = YOLO('yolov8n.pt')
model_custom = YOLO('xarm_wine_plate.pt')
names_base = model_base.names
names_custom = model_custom.names

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

print("[INFO] 相機初始化完成，按 e 拍照，按 q 離開...")

capture_count = 0
results_summary = {}

os.makedirs("captures", exist_ok=True)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        display_image = color_image.copy()

        # 預覽時畫上偵測框
        res_base = model_base(color_image)[0]
        res_custom = model_custom(color_image)[0]

        all_boxes = []
        boxes_base = [b for b in res_base.boxes if b.conf[0].item() > CONF_THRESHOLD]
        boxes_custom = [b for b in res_custom.boxes if b.conf[0].item() > CONF_THRESHOLD]

        for b in boxes_custom:
            all_boxes.append({'box': b, 'source': 'custom'})

        for b in boxes_base:
            x1b, y1b, x2b, y2b = map(int, b.xyxy[0].tolist())
            too_close = False
            for item in all_boxes:
                box_c = item['box']
                x1c, y1c, x2c, y2c = map(int, box_c.xyxy[0].tolist())
                iou = compute_iou([x1b, y1b, x2b, y2b], [x1c, y1c, x2c, y2c])
                if iou > IOU_THRESHOLD:
                    too_close = True
                    break
            if not too_close:
                all_boxes.append({'box': b, 'source': 'base'})

        for item in all_boxes:
            box = item['box']
            source = item['source']
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            name = names_custom.get(class_id, f"custom_{class_id}") if source == 'custom' else names_base.get(class_id, f"base_{class_id}")

            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, f"{name} ({conf:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("YOLO + Depth", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            print(f"[INFO] 拍照中...（第 {capture_count + 1} 次）")
            objects_detected = {}
            for item in all_boxes:
                box = item['box']
                source = item['source']
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                distance = get_distance(depth_frame, cx, cy)
                position = get_position(depth_frame, cx, cy, depth_intrin)

                name = names_custom.get(class_id, f"custom_{class_id}") if source == 'custom' else names_base.get(class_id, f"base_{class_id}")

                if name not in objects_detected:
                    objects_detected[name] = {
                        'distance': distance,
                        'confidence': round(conf, 3),
                        'position': position
                    }

                if name not in results_summary:
                    results_summary[name] = []
                results_summary[name].append(conf)

            json_filename = f'res{capture_count + 1}.json'
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(objects_detected, f, ensure_ascii=False, indent=2)

            print(f"[INFO] 儲存結果到 {json_filename}")
            capture_count += 1

        elif key == ord('q'):
            print("[INFO] 結束拍攝並輸出總結。")
            break

    final_summary = {}
    for name, confs in results_summary.items():
        avg_conf = round(sum(confs) / len(confs), 3)
        final_summary[name] = {
            'average_confidence': avg_conf,
            'counts': len(confs)
        }

    with open('summary.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 已儲存統整結果到 summary.json")
    print(f"[INFO] 總共拍攝了 {capture_count} 次照片。")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
