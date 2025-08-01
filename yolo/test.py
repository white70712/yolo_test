import pyrealsense2 as rs
import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

CONF_THRESHOLD = 0.4
TARGET_BOTTLES = ['Gin', 'Tequila', 'Whiskey']
WINE_COL_NAME = 'Wine column'

model = YOLO('yolo_wine.pt')
names = model.names

# ✅ bbox → 3D 座標
def get_3d_point_from_bbox(bbox, depth_frame, intrinsics):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    depth = depth_frame.get_distance(cx, cy)
    if depth == 0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

# ✅ 主流程：拍照、分類、排序、儲存 JSON
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
        # ⏱️ 等待相機穩定
        start_time = time.time()
        while time.time() - start_time < 5:
            pipeline.wait_for_frames()

        # ✅ 拍照
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
            conf = float(box.conf[0].item())
            name = names.get(class_id, f"class_{class_id}")
            pos3d = get_3d_point_from_bbox([x1, y1, x2, y2], depth_frame, intrinsics)
            if pos3d:
                detected_items.append({
                    "name": name,
                    "z": pos3d[2],
                    "Type": {
                        "id": None,
                        "name": name
                    },
                    "glass": name in TARGET_BOTTLES,
                    "wine col": name == WINE_COL_NAME
                })

        # ✅ 過濾出玻璃瓶，依 z 值排序（最遠 z 最小為 0，最近最大為 4）
        glass_items = [item for item in detected_items if item["glass"]]
        glass_items.sort(key=lambda i: i["z"])  # z 小在前，代表較遠
        for idx, item in enumerate(glass_items):
            item["Type"]["id"] = idx  # 0 ~ N，最遠編號 0

        # ✅ 合併所有項目（非 glass 的維持 id=None）
        final_output = glass_items + [i for i in detected_items if not i["glass"]]

        # ✅ 存成 JSON
        output_path = os.path.join(os.getcwd(), "final_wine_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        print(f"[INFO] 結果已儲存至 {output_path}")

    finally:
        pipeline.stop()
        print("[INFO] 相機已關閉")

# ✅ 執行
if __name__ == "__main__":
    detect_and_save_json()
