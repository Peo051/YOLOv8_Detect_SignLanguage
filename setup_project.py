"""
Script thiết lập project tự động

Cách sử dụng:
python setup_project.py
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def check_python_version():
    """Kiểm tra phiên bản Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠ Cần Python 3.8 trở lên")
        return False
    
    print("✓ Python version OK")
    return True

def install_requirements():
    """Cài đặt requirements"""
    if not os.path.exists("requirements.txt"):
        print("⚠ Không tìm thấy requirements.txt")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "CÀI ĐẶT DEPENDENCIES"
    )

def check_webcam():
    """Kiểm tra webcam"""
    print(f"\n{'='*60}")
    print("KIỂM TRA WEBCAM")
    print(f"{'='*60}\n")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            print("✓ Webcam hoạt động bình thường")
            ret, frame = cap.read()
            if ret:
                print(f"✓ Độ phân giải: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            return True
        else:
            print("⚠ Không mở được webcam")
            print("  - Kiểm tra webcam có kết nối không")
            print("  - Kiểm tra quyền truy cập camera")
            return False
    except Exception as e:
        print(f"⚠ Lỗi: {e}")
        return False

def check_model():
    """Kiểm tra model file"""
    print(f"\n{'='*60}")
    print("KIỂM TRA MODEL")
    print(f"{'='*60}\n")
    
    if os.path.exists("best.pt"):
        size_mb = os.path.getsize("best.pt") / (1024 * 1024)
        print(f"✓ Tìm thấy best.pt ({size_mb:.2f} MB)")
        return True
    else:
        print("⚠ Không tìm thấy best.pt")
        print("\nCách tải xuống:")
        print("1. Chạy: python download_from_drive.py")
        print("2. Hoặc tải thủ công từ:")
        print("   https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5")
        return False

def test_model():
    """Test model"""
    if not os.path.exists("best.pt"):
        return False
    
    print(f"\n{'='*60}")
    print("TEST MODEL")
    print(f"{'='*60}\n")
    
    try:
        from ultralytics import YOLO
        model = YOLO("best.pt")
        print(f"✓ Model tải thành công")
        print(f"✓ Số classes: {len(model.names)}")
        print(f"✓ Classes: {list(model.names.values())}")
        return True
    except Exception as e:
        print(f"⚠ Lỗi khi tải model: {e}")
        return False

def create_directories():
    """Tạo các thư mục cần thiết"""
    print(f"\n{'='*60}")
    print("TẠO THƯ MỤC")
    print(f"{'='*60}\n")
    
    dirs = ["captured_signs", "dataset/train", "dataset/val"]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}/")
    
    return True

def main():
    """Main function"""
    print("\n" + "="*60)
    print("THIẾT LẬP PROJECT NHẬN DIỆN NGÔN NGỮ KÝ HIỆU")
    print("="*60)
    
    steps = [
        ("Kiểm tra Python", check_python_version),
        ("Cài đặt dependencies", install_requirements),
        ("Tạo thư mục", create_directories),
        ("Kiểm tra webcam", check_webcam),
        ("Kiểm tra model", check_model),
        ("Test model", test_model),
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            print(f"⚠ Lỗi: {e}")
            results.append((step_name, False))
    
    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT")
    print("="*60 + "\n")
    
    for step_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "="*60)
        print("✓ THIẾT LẬP HOÀN TẤT!")
        print("="*60)
        print("\nBạn có thể chạy:")
        print("  python webcam_digits.py")
    else:
        print("\n" + "="*60)
        print("⚠ CÓ MỘT SỐ VẤN ĐỀ CẦN KHẮC PHỤC")
        print("="*60)
        print("\nVui lòng xem lại các bước bị lỗi ở trên")

if __name__ == "__main__":
    main()
