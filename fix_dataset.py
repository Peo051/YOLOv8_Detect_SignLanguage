"""
Script sửa cấu trúc dataset và train

Bạn có 2 dataset:
1. ASL Alphabet (A-Y): dataset/Dataset/dataset/
2. Sign Language Digits (0-9): dataset/Dataset/Sign-Language-Digits-Dataset-master/Dataset/
"""

import os
import shutil

print("="*60)
print("CHỌN DATASET ĐỂ TRAIN")
print("="*60)
print("\n1. ASL Alphabet (A-Y) - 24 classes, ~100 ảnh/class")
print("2. Sign Language Digits (0-9) - 10 classes")
print("3. Cả hai (kết hợp)")

choice = input("\nNhập lựa chọn (1/2/3): ").strip()

# Đường dẫn dataset gốc
ASL_ALPHABET_PATH = "dataset/Dataset/dataset"
DIGITS_PATH = "dataset/Dataset/Sign-Language-Digits-Dataset-master/Dataset"

# Đường dẫn dataset mới (đúng cấu trúc)
NEW_DATASET_PATH = "dataset_ready"

# Xóa dataset cũ nếu có
if os.path.exists(NEW_DATASET_PATH):
    shutil.rmtree(NEW_DATASET_PATH)

os.makedirs(f"{NEW_DATASET_PATH}/train", exist_ok=True)
os.makedirs(f"{NEW_DATASET_PATH}/val", exist_ok=True)

print(f"\n{'='*60}")
print("ĐANG TỔ CHỨC LẠI DATASET...")
print(f"{'='*60}\n")

def copy_dataset(source_path, dest_path, split_ratio=0.8):
    """Copy và split dataset thành train/val"""
    import random
    
    if not os.path.exists(source_path):
        print(f"⚠ Không tìm thấy: {source_path}")
        return 0, 0
    
    # Lấy danh sách classes
    classes = [d for d in os.listdir(source_path) 
               if os.path.isdir(os.path.join(source_path, d)) 
               and not d.startswith('.')]
    
    total_train = 0
    total_val = 0
    
    for cls in classes:
        cls_path = os.path.join(source_path, cls)
        
        # Lấy danh sách ảnh
        images = [f for f in os.listdir(cls_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
        
        # Shuffle và split
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy train
        train_cls_dir = os.path.join(dest_path, "train", cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        for img in train_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(train_cls_dir, img)
            shutil.copy2(src, dst)
        
        # Copy val
        val_cls_dir = os.path.join(dest_path, "val", cls)
        os.makedirs(val_cls_dir, exist_ok=True)
        for img in val_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(val_cls_dir, img)
            shutil.copy2(src, dst)
        
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"✓ {cls}: {len(train_images)} train, {len(val_images)} val")
    
    return total_train, total_val

# Copy dataset theo lựa chọn
if choice == "1":
    print("Đang copy ASL Alphabet dataset...\n")
    train_count, val_count = copy_dataset(ASL_ALPHABET_PATH, NEW_DATASET_PATH)
elif choice == "2":
    print("Đang copy Sign Language Digits dataset...\n")
    train_count, val_count = copy_dataset(DIGITS_PATH, NEW_DATASET_PATH)
elif choice == "3":
    print("Đang copy cả hai datasets...\n")
    print("1. ASL Alphabet:")
    train1, val1 = copy_dataset(ASL_ALPHABET_PATH, NEW_DATASET_PATH)
    print("\n2. Sign Language Digits:")
    train2, val2 = copy_dataset(DIGITS_PATH, NEW_DATASET_PATH)
    train_count = train1 + train2
    val_count = val1 + val2
else:
    print("⚠ Lựa chọn không hợp lệ")
    exit(1)

print(f"\n{'='*60}")
print("HOÀN TẤT TỔ CHỨC DATASET")
print(f"{'='*60}")
print(f"Tổng train: {train_count} ảnh")
print(f"Tổng val: {val_count} ảnh")
print(f"Dataset mới: {NEW_DATASET_PATH}/")

# Thống kê classes
train_classes = os.listdir(f"{NEW_DATASET_PATH}/train")
print(f"\nSố classes: {len(train_classes)}")
print(f"Classes: {sorted(train_classes)}")

print(f"\n{'='*60}")
print("BẮT ĐẦU TRAINING")
print(f"{'='*60}\n")

# Train
from ultralytics import YOLO
import torch

DEVICE = 'cpu' if not torch.cuda.is_available() else 0
MODEL_SIZE = "yolov8n-cls.pt"
EPOCHS = 50
BATCH_SIZE = 8 if DEVICE == 'cpu' else 32

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}\n")

if DEVICE == 'cpu':
    print("⚠ Đang dùng CPU - Training sẽ chậm")
    confirm = input("Tiếp tục? (y/n): ")
    if confirm.lower() != 'y':
        print("Đã hủy")
        exit(0)

model = YOLO(MODEL_SIZE)

results = model.train(
    data=NEW_DATASET_PATH,
    epochs=EPOCHS,
    imgsz=224,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=2,
    patience=15,
    save=True,
    plots=True,
    verbose=True,
    amp=False,
    cache=False,
)

print(f"\n{'='*60}")
print("✓ HOÀN THÀNH!")
print(f"{'='*60}")
print(f"Model: runs/classify/train/weights/best.pt")
print(f"\nCopy về thư mục gốc:")
print(f"  copy runs\\classify\\train\\weights\\best.pt best.pt")
print(f"{'='*60}\n")
