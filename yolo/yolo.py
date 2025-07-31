import pyrealsense2 as rs
import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

CONF_THRESHOLD = 0.4
TARGET_BOTTLES = ['Gin', 'Tequila', 'JOHN BARR Whiskey', 'JOHNNIE WALKER Whiskey', 'Rum']
WINE_COL_NAME = 'Wine column'

model = YOLO('yolo_wine.pt')
names = model.names

def get_3d_point_from_bbox(bbox, depth_frame, intrinsics):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    depth = depth_frame.get_distance(cx, cy)
    if depth == 0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

def detect_and_save_json():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    print("[INFO] 相機啟動中，等待中...")

    try:
        start_time = time.time()
        while time.time() - start_time < 5:
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("[ERROR] 拍照失敗")
            return

        color_image = np.asanyarray(color_frame.get_data())
        results = model(color_image)[0]
        boxes = [b for b in results.boxes if b.conf[0].item() > CONF_THRESHOLD]

        detected_items = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            name = names.get(class_id, f"class_{class_id}")
            pos3d = get_3d_point_from_bbox([x1, y1, x2, y2], depth_frame, intrinsics)
            if pos3d:
                detected_items.append({
                    "name": name,
                    "z": pos3d[2]
                })
            # 畫框與標籤
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(color_image,
                          (x1, y1 - text_height - baseline - 4),
                          (x1 + text_width + 2, y1),
                          (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(color_image, label, (x1 + 1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 針對酒瓶進行排序並給 id
        glass_items = [i for i in detected_items if i["name"] in TARGET_BOTTLES]
        glass_items.sort(key=lambda i: i["z"])  # z 越小越遠，排序是從最遠到最近
        type_entries = []
        for idx, item in enumerate(glass_items):
            type_entries.append({
                f"id_{idx}": {
                    "name": item["name"]
                }
            })


        # 判斷是否偵測到 Glass 與 Wine column
        status_flag = {
            "glass": any(i["name"].lower() == "glass" for i in detected_items),
            "wine col": any(i["name"] == WINE_COL_NAME for i in detected_items)
        }

        # 轉換格式為每個 key 一個獨立物件
        status_list = [{key: {"exist": value}} for key, value in status_flag.items()]

        final_output = type_entries + status_list

        # 儲存 JSON
        output_path = os.path.join(os.getcwd(), "final_wine_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        print(f"[INFO] 結果已儲存至 {output_path}")

    finally:
        pipeline.stop()
        print("[INFO] 相機已關閉")

if __name__ == "__main__":
    detect_and_save_json()
