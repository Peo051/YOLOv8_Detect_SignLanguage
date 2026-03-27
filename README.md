# Nhận Diện Ngôn Ngữ Ký Hiệu với CNN

Ứng dụng nhận diện ngôn ngữ ký hiệu real-time sử dụng YOLOv8 classification model.

![Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange.svg)

## 🎯 Tính năng

✅ Nhận diện real-time qua webcam  
✅ Hiển thị top 3 dự đoán với confidence scores  
✅ Thanh progress bar trực quan  
✅ Khung hướng dẫn đặt tay  
✅ Chụp và lưu ảnh kết quả  
✅ Hiển thị FPS  
✅ Cảnh báo khi confidence thấp  
✅ Giao diện thân thiện, dễ sử dụng

## 🚀 Cài đặt nhanh (Windows)

### Cách 1: Sử dụng batch file (Đơn giản nhất)

```bash
# Bước 1: Thiết lập
setup.bat

# Bước 2: Chạy ứng dụng
run.bat
```

### Cách 2: Thủ công

#### 1. Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

#### 2. Tải model từ Google Drive

**Tự động:**
```bash
python download_from_drive.py
```

**Thủ công:**
- Mở link: [Google Drive](https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5)
- Tải file `best.pt` về
- Đặt vào thư mục gốc của project

#### 3. Chạy ứng dụng

```bash
python webcam_digits.py
```

## ⌨️ Hướng dẫn sử dụng

| Phím | Chức năng |
|------|-----------|
| **ESC** | Thoát chương trình |
| **SPACE** | Chụp và lưu ảnh hiện tại |
| **H** | Bật/tắt khung hướng dẫn đặt tay |

## 📊 Kiểm tra hệ thống

Chạy script kiểm tra tự động:
```bash
python check_system.py
```

Hoặc thiết lập đầy đủ:
```bash
python setup_project.py
```

## 📁 Cấu trúc thư mục

```
project/
├── 🎯 Ứng dụng chính
│   ├── webcam_digits.py          # Ứng dụng nhận diện real-time
│   ├── best.pt                   # Model đã train (từ Drive)
│   └── config.py                 # File cấu hình
│
├── 🔧 Scripts hỗ trợ
│   ├── download_from_drive.py    # Tải model từ Google Drive
│   ├── setup_project.py          # Thiết lập tự động
│   ├── check_system.py           # Kiểm tra hệ thống
│   ├── collect_data.py           # Thu thập dữ liệu
│   ├── train_model.py            # Train model mới
│   └── test_model.py             # Test model trên ảnh
│
├── 📝 Hướng dẫn
│   ├── README.md                 # Hướng dẫn tổng quan
│   ├── QUICKSTART.md             # Hướng dẫn nhanh
│   └── HUONG_DAN.md              # Hướng dẫn chi tiết
│
├── 🪟 Windows batch files
│   ├── setup.bat                 # Thiết lập nhanh
│   └── run.bat                   # Chạy ứng dụng
│
├── 📦 Dependencies
│   └── requirements.txt
│
└── 📂 Thư mục dữ liệu (tự động tạo)
    ├── captured_signs/           # Ảnh chụp từ webcam
    └── dataset/                  # Dataset để train
        ├── train/
        └── val/
```

## Tùy chỉnh

Mở file `webcam_digits.py` và chỉnh sửa các tham số:

```python
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy tối thiểu
TOP_K = 3                   # Số lượng dự đoán hiển thị
SAVE_DIR = "captured_signs" # Thư mục lưu ảnh
```

## Xử lý lỗi thường gặp

### Lỗi: "Không mở được webcam"
- Kiểm tra webcam có hoạt động không
- Thử đổi `cv2.VideoCapture(0)` thành `cv2.VideoCapture(1)`
- Kiểm tra quyền truy cập camera

### Lỗi: "Không tìm thấy best.pt"
- Đảm bảo file `best.pt` nằm cùng thư mục với code
- Hoặc sửa đường dẫn trong biến `MODEL_PATH`

### Confidence thấp
- Cải thiện ánh sáng
- Đặt tay đúng vị trí trong khung hướng dẫn
- Đảm bảo background đơn giản

## Train model mới

Nếu muốn train lại model với dữ liệu riêng:

1. Chuẩn bị dataset theo cấu trúc:
```
dataset/
├── train/
│   ├── class_A/
│   ├── class_B/
│   └── ...
└── val/
    ├── class_A/
    ├── class_B/
    └── ...
```

2. Train với YOLOv8:
```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
model.train(data='dataset', epochs=50, imgsz=224)
```

## 📚 Tài liệu

- **Hướng dẫn nhanh**: [QUICKSTART.md](QUICKSTART.md)
- **Hướng dẫn chi tiết**: [HUONG_DAN.md](HUONG_DAN.md)
- **Google Drive**: [Dataset & Model](https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5)

## 🔗 Tham khảo

- Repo gốc: [SignLanguageDetectionUsingCNN](https://github.com/kumarvivek9088/SignLanguageDetectionUsingCNN)
- YOLOv8 Docs: https://docs.ultralytics.com/
- OpenCV Docs: https://docs.opencv.org/

## 🎓 Các lệnh hữu ích

```bash
# Chạy ứng dụng
python webcam_digits.py

# Test model trên ảnh
python test_model.py --image test.jpg --save

# Test model trên thư mục
python test_model.py --folder test_images/ --save

# Thu thập dữ liệu mới
python collect_data.py

# Train model mới
python train_model.py

# Kiểm tra hệ thống
python check_system.py
```

## 🐛 Xử lý lỗi

| Lỗi | Giải pháp |
|-----|-----------|
| "Không tìm thấy best.pt" | `python download_from_drive.py` |
| "Không mở được webcam" | Đổi `CAMERA_INDEX = 0` thành `1` trong config.py |
| "No module named..." | `pip install -r requirements.txt` |
| "Invalid CUDA device" | Máy không có GPU. Dùng `python train_model_cpu.py` hoặc Google Colab |
| Confidence thấp | Cải thiện ánh sáng, đặt tay trong khung |

## 🎓 Training model mới

Nếu muốn train lại model với dữ liệu riêng:

### Cách 1: Google Colab (Khuyến nghị - GPU miễn phí)
1. Upload file `train_on_colab.ipynb` lên [Google Colab](https://colab.research.google.com/)
2. Bật GPU: Runtime > Change runtime type > GPU
3. Làm theo hướng dẫn trong notebook

### Cách 2: Trên máy local
```bash
# Nếu có GPU
python train_model.py

# Nếu chỉ có CPU (chậm)
python train_model_cpu.py
```

Xem hướng dẫn chi tiết: [TRAIN_GUIDE.md](TRAIN_GUIDE.md)

## 📄 License

MIT License
