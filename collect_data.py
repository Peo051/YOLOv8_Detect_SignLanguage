"""
Script thu thập dữ liệu từ webcam để train model

Cách sử dụng:
1. Chạy: python collect_data.py
2. Nhập tên class (ví dụ: A, B, C, ...)
3. Nhấn SPACE để chụp ảnh
4. Nhấn N để chuyển sang class khác
5. Nhấn ESC để thoát
"""

import cv2
import os
from datetime import datetime

# ==================== CẤU HÌNH ====================
DATASET_PATH = "dataset/train"
IMAGES_PER_CLASS = 100  # Số ảnh khuyến nghị mỗi class
IMAGE_SIZE = (224, 224)

# ==================== KHỞI TẠO ====================
os.makedirs(DATASET_PATH, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Không mở được webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==================== HÀM HỖ TRỢ ====================
def draw_ui(frame, class_name, count, target):
    """Vẽ giao diện hướng dẫn"""
    h, w = frame.shape[:2]
    
    # Background overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Thông tin
    cv2.putText(frame, f"Class: {class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Da chup: {count}/{target}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Progress bar
    progress = min(count / target, 1.0)
    bar_width = int(progress * (w - 20))
    cv2.rectangle(frame, (10, 80), (10 + bar_width, 100), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, 80), (w - 10, 100), (255, 255, 255), 2)
    
    # Hướng dẫn
    cv2.putText(frame, "SPACE: Chup | N: Class moi | ESC: Thoat", 
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Khung hướng dẫn
    center_x, center_y = w // 2, h // 2
    box_size = 250
    x1 = center_x - box_size // 2
    y1 = center_y - box_size // 2
    x2 = center_x + box_size // 2
    y2 = center_y + box_size // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    return frame

def save_image(frame, class_name, count):
    """Lưu ảnh vào thư mục class"""
    class_dir = os.path.join(DATASET_PATH, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{class_dir}/{class_name}_{count:04d}_{timestamp}.jpg"
    
    # Crop vùng trung tâm và resize
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    box_size = 250
    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(w, center_x + box_size // 2)
    y2 = min(h, center_y + box_size // 2)
    
    cropped = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped, IMAGE_SIZE)
    
    cv2.imwrite(filename, resized)
    print(f"✓ Đã lưu: {filename}")
    return filename

# ==================== MAIN ====================
print("\n=== THU THẬP DỮ LIỆU ===")
print(f"Thư mục lưu: {DATASET_PATH}")
print(f"Khuyến nghị: {IMAGES_PER_CLASS} ảnh/class")
print("========================\n")

current_class = input("Nhập tên class (ví dụ: A, B, C, ...): ").strip()
if not current_class:
    current_class = "Unknown"

count = 0

# Đếm số ảnh đã có
class_dir = os.path.join(DATASET_PATH, current_class)
if os.path.exists(class_dir):
    existing = len([f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"⚠ Class '{current_class}' đã có {existing} ảnh")
    count = existing

print(f"\n✓ Bắt đầu thu thập cho class: {current_class}")
print("Đặt tay vào khung màu vàng và nhấn SPACE để chụp\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame = draw_ui(frame, current_class, count, IMAGES_PER_CLASS)
    
    cv2.imshow("Thu thap du lieu", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE - chụp ảnh
        save_image(frame, current_class, count)
        count += 1
        
        if count >= IMAGES_PER_CLASS:
            print(f"\n✓ Đã đủ {IMAGES_PER_CLASS} ảnh cho class '{current_class}'")
            print("Nhấn N để chuyển class mới hoặc ESC để thoát")
    
    elif key == ord('n') or key == ord('N'):  # N - class mới
        new_class = input("\nNhập tên class mới: ").strip()
        if new_class:
            current_class = new_class
            count = 0
            
            # Đếm số ảnh đã có
            class_dir = os.path.join(DATASET_PATH, current_class)
            if os.path.exists(class_dir):
                existing = len([f for f in os.listdir(class_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"⚠ Class '{current_class}' đã có {existing} ảnh")
                count = existing
            
            print(f"✓ Chuyển sang class: {current_class}\n")

cap.release()
cv2.destroyAllWindows()

# ==================== THỐNG KÊ ====================
print("\n=== THỐNG KÊ DỮ LIỆU ===")
classes = [d for d in os.listdir(DATASET_PATH) 
           if os.path.isdir(os.path.join(DATASET_PATH, d))]

total = 0
for cls in sorted(classes):
    cls_path = os.path.join(DATASET_PATH, cls)
    num_images = len([f for f in os.listdir(cls_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total += num_images
    status = "✓" if num_images >= IMAGES_PER_CLASS else "⚠"
    print(f"{status} {cls}: {num_images} ảnh")

print(f"\nTổng: {total} ảnh, {len(classes)} classes")
print("========================\n")
