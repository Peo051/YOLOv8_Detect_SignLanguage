# HƯỚNG DẪN NHANH

## Bắt đầu trong 3 bước

### Bước 1: Cài đặt

```bash
pip install -r requirements.txt
```

### Bước 2: Tải model từ Google Drive

**Cách 1: Tự động (khuyến nghị)**
```bash
python download_from_drive.py
```

**Cách 2: Thủ công**
1. Mở link: https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5
2. Tải file `best.pt` về
3. Đặt file `best.pt` vào thư mục gốc của project

### Bước 3: Chạy ứng dụng

```bash
python webcam_digits.py
```

## Phím tắt

- **ESC**: Thoát chương trình
- **SPACE**: Chụp và lưu ảnh
- **H**: Bật/tắt khung hướng dẫn

## Kiểm tra thiết lập

Chạy script kiểm tra tự động:
```bash
python setup_project.py
```

Script này sẽ kiểm tra:
- ✓ Python version
- ✓ Dependencies đã cài đặt
- ✓ Webcam hoạt động
- ✓ Model file tồn tại
- ✓ Model load được

## Cấu trúc thư mục

```
project/
├── webcam_digits.py          # Ứng dụng chính ⭐
├── best.pt                   # Model đã train (tải từ Drive)
├── requirements.txt          # Dependencies
├── download_from_drive.py    # Script tải từ Drive
├── setup_project.py          # Script kiểm tra thiết lập
├── collect_data.py           # Thu thập dữ liệu mới
├── train_model.py            # Train model mới
├── test_model.py             # Test model trên ảnh
├── config.py                 # Cấu hình
├── captured_signs/           # Ảnh chụp (tự động tạo)
└── dataset/                  # Dataset (nếu muốn train lại)
    ├── train/
    └── val/
```

## Các lệnh hữu ích

### Chạy ứng dụng
```bash
python webcam_digits.py
```

### Test model trên ảnh
```bash
python test_model.py --image test.jpg --save
```

### Test model trên thư mục
```bash
python test_model.py --folder test_images/ --save
```

### Thu thập dữ liệu mới
```bash
python collect_data.py
```

### Train model mới
```bash
python train_model.py
```

## Xử lý lỗi nhanh

### "Không tìm thấy best.pt"
```bash
python download_from_drive.py
```

### "Không mở được webcam"
Sửa trong `webcam_digits.py`:
```python
CAMERA_INDEX = 0  # Đổi thành 1 hoặc 2
```

### "No module named 'ultralytics'"
```bash
pip install -r requirements.txt
```

### Confidence thấp
- Cải thiện ánh sáng
- Đặt tay trong khung hướng dẫn
- Đảm bảo background đơn giản

## Tùy chỉnh nhanh

Mở `webcam_digits.py` và sửa:

```python
# Dòng 8-10
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy (0.3-0.8)
TOP_K = 3                   # Số dự đoán hiển thị (1-5)
SAVE_DIR = "captured_signs" # Thư mục lưu ảnh
```

## Link tham khảo

- Google Drive: https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5
- Repo gốc: https://github.com/kumarvivek9088/SignLanguageDetectionUsingCNN
- YOLOv8 Docs: https://docs.ultralytics.com/

## Cần trợ giúp?

Xem hướng dẫn chi tiết: `HUONG_DAN.md`
