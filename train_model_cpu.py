"""
Script train model tối ưu cho CPU (không có GPU)

Cách sử dụng:
1. Chuẩn bị dataset theo cấu trúc:
   dataset/
   ├── train/
   │   ├── A/
   │   ├── B/
   │   └── ...
   └── val/
       ├── A/
       ├── B/
       └── ...

2. Chạy: python train_model_cpu.py
"""

from ultralytics import YOLO
import os
import torch

# ==================== CẤU HÌNH TỐI ƯU CHO CPU ====================
DATASET_PATH = "dataset"
MODEL_SIZE = "yolov8n-cls.pt"  # Dùng nano (nhỏ nhất) cho CPU
EPOCHS = 50  # Giảm epochs cho CPU
IMAGE_SIZE = 224
BATCH_SIZE = 8  # Giảm batch size cho CPU
DEVICE = 'cpu'
WORKERS = 2  # Số workers cho data loading

print(f"\n{'='*60}")
print(f"TRAINING TỐI ƯU CHO CPU")
print(f"{'='*60}")
print(f"⚠ Lưu ý: Training trên CPU sẽ chậm hơn GPU rất nhiều")
print(f"  - Dự kiến: 5-10 phút/epoch (tùy số lượng dữ liệu)")
print(f"  - Khuyến nghị: Giảm số epochs hoặc dùng Google Colab (GPU miễn phí)")
print(f"{'='*60}\n")

# ==================== KIỂM TRA DATASET ====================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {DATASET_PATH}")

train_path = os.path.join(DATASET_PATH, "train")
val_path = os.path.join(DATASET_PATH, "val")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_path}")

# Đếm số lượng classes
classes = [d for d in os.listdir(train_path) 
           if os.path.isdir(os.path.join(train_path, d))]
print(f"✓ Tìm thấy {len(classes)} classes: {classes}")

# Đếm số lượng ảnh
total_images = 0
for cls in classes:
    cls_path = os.path.join(train_path, cls)
    num_images = len([f for f in os.listdir(cls_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  - {cls}: {num_images} ảnh")
    total_images += num_images

print(f"\n✓ Tổng số ảnh training: {total_images}")

# Ước tính thời gian
estimated_time_per_epoch = (total_images / BATCH_SIZE) * 0.5  # ~0.5s per batch trên CPU
estimated_total_minutes = (estimated_time_per_epoch * EPOCHS) / 60

print(f"\n⏱ Ước tính thời gian:")
print(f"  - {estimated_time_per_epoch:.1f}s/epoch")
print(f"  - Tổng: ~{estimated_total_minutes:.1f} phút ({estimated_total_minutes/60:.1f} giờ)")

# Xác nhận
print(f"\n{'='*60}")
response = input("Bạn có muốn tiếp tục training? (y/n): ")
if response.lower() != 'y':
    print("Đã hủy training")
    exit(0)

# ==================== TRAIN MODEL ====================
print(f"\n{'='*60}")
print(f"BẮT ĐẦU TRAINING")
print(f"{'='*60}")
print(f"Model: {MODEL_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: CPU")
print(f"Workers: {WORKERS}")
print(f"{'='*60}\n")

# Load pretrained model
model = YOLO(MODEL_SIZE)

# Train với cấu hình tối ưu cho CPU
results = model.train(
    data=DATASET_PATH,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,
    patience=15,  # Early stopping sớm hơn
    save=True,
    plots=True,
    verbose=True,
    # Tối ưu cho CPU
    amp=False,  # Tắt automatic mixed precision
    cache=False,  # Không cache để tiết kiệm RAM
)

print(f"\n{'='*60}")
print(f"✓ HOÀN THÀNH TRAINING")
print(f"{'='*60}")
print(f"Model đã lưu tại: runs/classify/train/weights/best.pt")
print(f"Copy file best.pt về thư mục gốc để sử dụng với webcam_digits.py")
print(f"{'='*60}\n")

# ==================== ĐÁNH GIÁ MODEL ====================
print("Đang đánh giá model trên validation set...")
metrics = model.val()

print(f"\nKết quả:")
print(f"  - Top-1 Accuracy: {metrics.top1:.2%}")
print(f"  - Top-5 Accuracy: {metrics.top5:.2%}")

# ==================== HƯỚNG DẪN TIẾP THEO ====================
print(f"\n{'='*60}")
print(f"HƯỚNG DẪN TIẾP THEO")
print(f"{'='*60}")
print(f"1. Copy model về thư mục gốc:")
print(f"   copy runs\\classify\\train\\weights\\best.pt best.pt")
print(f"\n2. Test model:")
print(f"   python test_model.py --folder dataset/val --save")
print(f"\n3. Chạy ứng dụng:")
print(f"   python webcam_digits.py")
print(f"{'='*60}\n")
