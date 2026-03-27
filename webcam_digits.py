from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime

# ==================== CẤU HÌNH ====================
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy tối thiểu
TOP_K = 3  # Hiển thị top 3 dự đoán
SAVE_DIR = "captured_signs"  # Thư mục lưu ảnh chụp

# ==================== KHỞI TẠO ====================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy {MODEL_PATH}. Hãy đặt file best.pt cùng thư mục!")

os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
print(f"✓ Đã tải model: {MODEL_PATH}")
print(f"✓ Các lớp nhận diện: {list(model.names.values())}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Không mở được webcam. Thử đổi 0 -> 1 hoặc kiểm tra quyền camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==================== HÀM HỖ TRỢ ====================
def draw_info_panel(frame, predictions, fps):
    """Vẽ bảng thông tin với top predictions"""
    h, w = frame.shape[:2]
    panel_height = 180
    
    # Tạo overlay trong suốt
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Tiêu đề
    cv2.putText(frame, "NHAN DIEN NGON NGU KY HIEU", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Top predictions
    y_offset = 70
    for i, (name, conf) in enumerate(predictions[:TOP_K]):
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        bar_width = int(conf * 300)
        
        # Tên và confidence
        text = f"{i+1}. {name}: {conf:.2%}"
        cv2.putText(frame, text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Thanh progress
        cv2.rectangle(frame, (10, y_offset + 5), (10 + bar_width, y_offset + 15), color, -1)
        
        y_offset += 35
    
    return frame

def draw_hand_guide(frame):
    """Vẽ khung hướng dẫn đặt tay"""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    box_size = 250
    
    x1 = center_x - box_size // 2
    y1 = center_y - box_size // 2
    x2 = center_x + box_size // 2
    y2 = center_y + box_size // 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, "Dat tay vao day", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return frame

def save_capture(frame, prediction, confidence):
    """Lưu ảnh chụp với timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/{prediction}_{confidence:.2f}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✓ Đã lưu: {filename}")
    return filename

# ==================== VÒNG LẶP CHÍNH ====================
print("\n=== HƯỚNG DẪN SỬ DỤNG ===")
print("ESC    : Thoát chương trình")
print("SPACE  : Chụp và lưu ảnh")
print("H      : Bật/tắt khung hướng dẫn")
print("========================\n")

show_guide = True
frame_count = 0
fps = 0
prev_time = cv2.getTickCount()

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠ Không đọc được frame từ webcam")
        break
    
    frame = cv2.flip(frame, 1)  # Lật ngang để dễ sử dụng
    
    # Dự đoán
    results = model.predict(source=frame, imgsz=224, verbose=False)[0]
    
    # Lấy top predictions
    probs = results.probs.data.cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    predictions = [(model.names[idx], float(probs[idx])) for idx in top_indices]
    
    top_name, top_conf = predictions[0]
    
    # Tính FPS
    frame_count += 1
    if frame_count % 10 == 0:
        curr_time = cv2.getTickCount()
        fps = 10 / ((curr_time - prev_time) / cv2.getTickFrequency())
        prev_time = curr_time
    
    # Vẽ UI
    if show_guide:
        frame = draw_hand_guide(frame)
    
    frame = draw_info_panel(frame, predictions, fps)
    
    # Cảnh báo nếu confidence thấp
    if top_conf < CONFIDENCE_THRESHOLD:
        cv2.putText(frame, "! Confidence thap - Kiem tra anh sang", 
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Sign Language Detection", frame)
    
    # Xử lý phím
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE - chụp ảnh
        save_capture(frame, top_name, top_conf)
    elif key == ord('h') or key == ord('H'):  # H - toggle guide
        show_guide = not show_guide

cap.release()
cv2.destroyAllWindows()
print("\n✓ Đã đóng chương trình")