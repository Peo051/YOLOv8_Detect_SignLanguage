"""
Script tải xuống data và model từ Google Drive

Cách sử dụng:
1. Cài đặt gdown: pip install gdown
2. Chạy: python download_from_drive.py
"""

import os
import sys

try:
    import gdown
except ImportError:
    print("⚠ Chưa cài đặt gdown. Đang cài đặt...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

# ==================== CẤU HÌNH ====================
# Link Google Drive folder của bạn
DRIVE_FOLDER_ID = "1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5"
DOWNLOAD_DIR = "downloaded_data"

# ==================== TẢI XUỐNG ====================
print("="*60)
print("TẢI XUỐNG DỮ LIỆU TỪ GOOGLE DRIVE")
print("="*60)

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Tải toàn bộ folder
folder_url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
print(f"\nĐang tải từ: {folder_url}")
print(f"Lưu vào: {DOWNLOAD_DIR}/\n")

try:
    gdown.download_folder(
        url=folder_url,
        output=DOWNLOAD_DIR,
        quiet=False,
        use_cookies=False
    )
    print("\n✓ Tải xuống thành công!")
    
except Exception as e:
    print(f"\n⚠ Lỗi khi tải: {e}")
    print("\nGiải pháp thay thế:")
    print("1. Mở link: https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5")
    print("2. Nhấn 'Download' để tải về máy")
    print("3. Giải nén vào thư mục dự án")
    sys.exit(1)

# ==================== TỔ CHỨC LẠI FILE ====================
print("\n" + "="*60)
print("TỔ CHỨC LẠI FILE")
print("="*60)

# Tìm file best.pt
for root, dirs, files in os.walk(DOWNLOAD_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        
        # Copy best.pt về thư mục gốc
        if file == "best.pt":
            import shutil
            dest = "best.pt"
            shutil.copy2(file_path, dest)
            print(f"✓ Đã copy {file} -> {dest}")
        
        # Copy dataset
        elif "dataset" in root.lower() or "data" in root.lower():
            print(f"  Tìm thấy: {file_path}")

print("\n" + "="*60)
print("HOÀN TẤT")
print("="*60)
print("\nBạn có thể:")
print("1. Chạy ứng dụng: python webcam_digits.py")
print("2. Test model: python test_model.py --folder downloaded_data/test")
print("3. Train lại: python train_model.py")
