"""
Script để train model YOLOv8 classification cho nhận diện ngôn ngữ ký hiệu

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

2. Chạy: python train_model.py
"""

from ultralytics import YOLO
import os

# ==================== CẤU HÌNH ====================
DATASET_PATH = "dataset"  # Đường dẫn đến thư mục dataset
MODEL_SIZE = "yolov8n-cls.pt"  # n=nano, s=small, m=medium, l=large, x=xlarge
EPOCHS = 100
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Tự động phát hiện device (GPU hoặc CPU)
import torch
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE} ({'GPU' if torch.cuda.is_available() else 'CPU'})")

# ==================== KIỂM TRA DATASET ====================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {DATASET_PATH}")

train_path = os.path.join(DATASET_PATH, "train")
val_path = os.path.join(DATASET_PATH, "val")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_path}")

if not os.path.exists(val_path):
    print(f"⚠ Không tìm thấy thư mục val: {val_path}")
    print("Sẽ tự động split từ train set")

# Đếm số lượng classes
classes = [d for d in os.listdir(train_path) 
           if os.path.isdir(os.path.join(train_path, d))]
print(f"\n✓ Tìm thấy {len(classes)} classes: {classes}")

# Đếm số lượng ảnh
total_images = 0
for cls in classes:
    cls_path = os.path.join(train_path, cls)
    num_images = len([f for f in os.listdir(cls_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  - {cls}: {num_images} ảnh")
    total_images += num_images

print(f"\n✓ Tổng số ảnh training: {total_images}")

# ==================== TRAIN MODEL ====================
print(f"\n{'='*50}")
print(f"BẮT ĐẦU TRAINING")
print(f"{'='*50}")
print(f"Model: {MODEL_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {DEVICE} ({'GPU - Nhanh' if DEVICE != 'cpu' else 'CPU - Chậm hơn'})")
if DEVICE == 'cpu':
    print(f"⚠ Đang dùng CPU. Training sẽ chậm hơn nhiều.")
    print(f"  Để dùng GPU, cài đặt PyTorch với CUDA:")
    print(f"  https://pytorch.org/get-started/locally/")
print(f"{'='*50}\n")

# Load pretrained model
model = YOLO(MODEL_SIZE)

# Train
results = model.train(
    data=DATASET_PATH,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    patience=20,  # Early stopping
    save=True,
    plots=True,
    verbose=True
)

print(f"\n{'='*50}")
print(f"✓ HOÀN THÀNH TRAINING")
print(f"{'='*50}")
print(f"Model đã lưu tại: runs/classify/train/weights/best.pt")
print(f"Copy file best.pt về thư mục gốc để sử dụng với webcam_digits.py")
print(f"{'='*50}\n")

# ==================== ĐÁNH GIÁ MODEL ====================
print("Đang đánh giá model trên validation set...")
metrics = model.val()

print(f"\nKết quả:")
print(f"  - Top-1 Accuracy: {metrics.top1:.2%}")
print(f"  - Top-5 Accuracy: {metrics.top5:.2%}")
