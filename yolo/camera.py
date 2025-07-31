import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

IOU_THRESHOLD = 0.5  # 降低一點試試
CONF_THRESHOLD = 0.4

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

model_base = YOLO('yolov8n.pt')
model_custom = YOLO('xarm_test.pt')
names_base = model_base.names
names_custom = model_custom.names

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

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

        res_base = model_base(color_image)[0]
        res_custom = model_custom(color_image)[0]

        boxes_base = [b for b in res_base.boxes if b.conf[0].item() > CONF_THRESHOLD]
        boxes_custom = [b for b in res_custom.boxes if b.conf[0].item() > CONF_THRESHOLD]

        all_boxes = []
        # 先加入 custom
        for b in boxes_custom:
            all_boxes.append({'box': b, 'source': 'custom'})

        # 加入不重疊 base
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

            if source == 'custom':
                name = names_custom.get(class_id, f"custom_{class_id}")
            else:
                name = names_base.get(class_id, f"base_{class_id}")

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({conf:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(color_image,
                          (x1, y1 - text_height - baseline - 4),
                          (x1 + text_width + 2, y1),
                          (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(color_image, label, (x1 + 1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("YOLO + Depth", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
