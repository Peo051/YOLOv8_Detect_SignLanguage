"""
Script kiểm tra nhanh hệ thống

Cách sử dụng: python check_system.py
"""

import sys
import os

def check_item(name, check_func):
    """Kiểm tra một mục"""
    try:
        result = check_func()
        if result:
            print(f"✓ {name}")
            return True
        else:
            print(f"✗ {name}")
            return False
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

print("\n" + "="*60)
print("KIỂM TRA HỆ THỐNG")
print("="*60 + "\n")

checks = []

# Python version
def check_python():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  Python {version.major}.{version.minor}.{version.micro}")
        return True
    return False

checks.append(check_item("Python 3.8+", check_python))

# OpenCV
def check_opencv():
    import cv2
    print(f"  OpenCV {cv2.__version__}")
    return True

checks.append(check_item("OpenCV", check_opencv))

# Ultralytics
def check_ultralytics():
    import ultralytics
    print(f"  Ultralytics {ultralytics.__version__}")
    return True

checks.append(check_item("Ultralytics", check_ultralytics))

# NumPy
def check_numpy():
    import numpy as np
    print(f"  NumPy {np.__version__}")
    return True

checks.append(check_item("NumPy", check_numpy))

# Webcam
def check_webcam():
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
        return True
    return False

checks.append(check_item("Webcam", check_webcam))

# Model file
def check_model_file():
    if os.path.exists("best.pt"):
        size_mb = os.path.getsize("best.pt") / (1024 * 1024)
        print(f"  {size_mb:.2f} MB")
        return True
    return False

checks.append(check_item("Model file (best.pt)", check_model_file))

# Model loading
def check_model_load():
    if not os.path.exists("best.pt"):
        return False
    from ultralytics import YOLO
    model = YOLO("best.pt")
    print(f"  {len(model.names)} classes")
    return True

checks.append(check_item("Model loading", check_model_load))

# GPU (optional)
def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  CPU only")
            return True
    except:
        print("  CPU only")
        return True

checks.append(check_item("GPU/CPU", check_gpu))

# Tổng kết
print("\n" + "="*60)
passed = sum(checks)
total = len(checks)

if passed == total:
    print(f"✓ TẤT CẢ OK ({passed}/{total})")
    print("="*60)
    print("\nBạn có thể chạy: python webcam_digits.py")
else:
    print(f"⚠ CÓ VẤN ĐỀ ({passed}/{total})")
    print("="*60)
    
    if not checks[5]:  # Model file
        print("\nCần tải model:")
        print("  python download_from_drive.py")
    
    if not checks[0] or not checks[1] or not checks[2] or not checks[3]:
        print("\nCần cài đặt dependencies:")
        print("  pip install -r requirements.txt")
    
    if not checks[4]:  # Webcam
        print("\nKiểm tra webcam:")
        print("  - Webcam có kết nối không?")
        print("  - Quyền truy cập camera?")

print()
