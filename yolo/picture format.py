import os
from PIL import Image
import pillow_heif

input_folder = r"C:\Users\user\Desktop\soda"
output_format = "jpg"  # 可改為 "png"

def convert_all_heic_in_folder(folder, out_format="jpg"):
    for filename in os.listdir(folder):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(folder, filename)
            try:
                heif_file = pillow_heif.read_heif(heic_path)
                image = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data
                )
                out_path = os.path.splitext(heic_path)[0] + f".{out_format}"
                save_format = "JPEG" if out_format.lower() == "jpg" else out_format.upper()
                image.save(out_path, format=save_format)
                print(f"✅ 轉換完成: {filename} → {os.path.basename(out_path)}")
            except Exception as e:
                print(f"❌ 轉換失敗: {filename}，錯誤：{e}")

convert_all_heic_in_folder(input_folder, output_format)
