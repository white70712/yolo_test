import cv2
import os

video_path = 'train.mp4'
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0
interval = 3  # 每3幀存一張

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        filename = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f'Total {saved_count} frames saved.')
