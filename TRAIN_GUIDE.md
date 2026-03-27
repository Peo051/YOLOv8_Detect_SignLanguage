# HƯỚNG DẪN TRAINING MODEL

## ⚠ Quan trọng: GPU vs CPU

Training model deep learning trên CPU rất chậm. Bạn có 3 lựa chọn:

### 1. 🚀 Google Colab (Khuyến nghị - GPU miễn phí)

**Ưu điểm:**
- GPU miễn phí (Tesla T4)
- Nhanh hơn CPU 10-50 lần
- Không cần cài đặt gì

**Cách sử dụng:**
1. Mở [Google Colab](https://colab.research.google.com/)
2. Upload file `train_on_colab.ipynb`
3. Bật GPU: `Runtime` > `Change runtime type` > `Hardware accelerator` > `GPU`
4. Upload dataset lên Google Drive
5. Chạy từng cell theo thứ tự

**Thời gian ước tính:**
- 100 ảnh/class, 10 classes, 50 epochs: ~15-30 phút

### 2. 💻 Training trên máy có GPU

Nếu máy bạn có GPU NVIDIA:

```bash
# Cài đặt PyTorch với CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"

# Train
python train_model.py
```

### 3. 🐌 Training trên CPU (Chậm)

Nếu không có GPU và không dùng Colab:

```bash
python train_model_cpu.py
```

**Lưu ý:**
- Rất chậm (có thể mất vài giờ đến vài ngày)
- Chỉ nên dùng cho dataset nhỏ (<500 ảnh tổng)
- Giảm epochs xuống 20-30

## 📊 So sánh thời gian

| Phương pháp | Dataset nhỏ (500 ảnh) | Dataset vừa (2000 ảnh) | Dataset lớn (5000 ảnh) |
|-------------|----------------------|----------------------|----------------------|
| **Google Colab GPU** | 10-15 phút | 30-45 phút | 1-2 giờ |
| **GPU RTX 3060** | 5-10 phút | 20-30 phút | 45-90 phút |
| **CPU (i5/i7)** | 2-4 giờ | 8-12 giờ | 1-2 ngày |

## 🎯 Chuẩn bị dataset

### Cấu trúc thư mục

```
dataset/
├── train/
│   ├── A/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── B/
│   ├── C/
│   └── ...
└── val/  (tùy chọn, sẽ tự động split nếu không có)
    ├── A/
    ├── B/
    └── ...
```

### Thu thập dữ liệu

```bash
python collect_data.py
```

**Khuyến nghị:**
- Ít nhất 100-200 ảnh/class
- Đa dạng góc độ, ánh sáng, background
- Ảnh rõ nét, không bị mờ

## ⚙️ Cấu hình training

### Chỉnh sửa trong `train_model.py`:

```python
# Model size
MODEL_SIZE = "yolov8n-cls.pt"  # n=nhanh, s=cân bằng, m/l=chính xác

# Training parameters
EPOCHS = 100        # Số epochs (50-100 cho bắt đầu)
BATCH_SIZE = 32     # Batch size (giảm nếu hết RAM)
IMAGE_SIZE = 224    # Kích thước ảnh input
```

### Chọn model size:

| Model | Tốc độ | Độ chính xác | Kích thước | Khuyến nghị |
|-------|--------|--------------|------------|-------------|
| yolov8n-cls | Rất nhanh | Tốt | 5 MB | Webcam real-time |
| yolov8s-cls | Nhanh | Tốt hơn | 11 MB | Cân bằng |
| yolov8m-cls | Trung bình | Cao | 26 MB | Độ chính xác cao |
| yolov8l-cls | Chậm | Rất cao | 44 MB | Production |

## 🚀 Bắt đầu training

### Trên Google Colab (Khuyến nghị):

1. Upload `train_on_colab.ipynb` lên Colab
2. Bật GPU
3. Chạy từng cell

### Trên máy local:

```bash
# Kiểm tra GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Train với GPU (nếu có)
python train_model.py

# Train với CPU (chậm)
python train_model_cpu.py
```

## 📈 Theo dõi training

Training sẽ hiển thị:
- Loss (giảm dần là tốt)
- Accuracy (tăng dần là tốt)
- Thời gian còn lại

Kết quả lưu tại: `runs/classify/train/`

## ✅ Sau khi training

### 1. Copy model về thư mục gốc

```bash
# Windows
copy runs\classify\train\weights\best.pt best.pt

# Linux/Mac
cp runs/classify/train/weights/best.pt best.pt
```

### 2. Test model

```bash
python test_model.py --folder dataset/val --save
```

### 3. Chạy ứng dụng

```bash
python webcam_digits.py
```

## 🔧 Xử lý lỗi

### Lỗi: "CUDA out of memory"

Giảm batch size:
```python
BATCH_SIZE = 16  # hoặc 8
```

### Lỗi: "Invalid CUDA device"

Máy không có GPU, dùng CPU:
```python
DEVICE = 'cpu'
```

Hoặc dùng Google Colab.

### Training quá chậm

- Dùng Google Colab (GPU miễn phí)
- Giảm số epochs: `EPOCHS = 30`
- Dùng model nhỏ hơn: `MODEL_SIZE = "yolov8n-cls.pt"`
- Giảm kích thước dataset

### Accuracy thấp

- Tăng số lượng ảnh training (>200/class)
- Cải thiện chất lượng ảnh
- Tăng epochs: `EPOCHS = 150`
- Dùng model lớn hơn: `MODEL_SIZE = "yolov8s-cls.pt"`
- Kiểm tra data augmentation

## 💡 Tips để cải thiện model

1. **Dữ liệu chất lượng cao**
   - Ảnh rõ nét, đủ sáng
   - Đa dạng góc độ
   - Background đơn giản

2. **Đủ số lượng**
   - Tối thiểu 100 ảnh/class
   - Khuyến nghị 200-500 ảnh/class

3. **Cân bằng classes**
   - Số lượng ảnh mỗi class nên tương đương nhau

4. **Data augmentation**
   - YOLOv8 tự động áp dụng
   - Flip, rotate, brightness, contrast

5. **Early stopping**
   - Model tự động dừng nếu không cải thiện
   - `patience=20` (dừng sau 20 epochs không cải thiện)

## 📚 Tài liệu tham khảo

- [YOLOv8 Classification](https://docs.ultralytics.com/tasks/classify/)
- [Google Colab](https://colab.research.google.com/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
