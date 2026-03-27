# HƯỚNG DẪN CHI TIẾT

## Quy trình hoàn chỉnh từ đầu đến cuối

### Bước 1: Cài đặt môi trường

```bash
# Cài đặt Python packages
pip install -r requirements.txt

# Kiểm tra cài đặt
python -c "import cv2; import ultralytics; print('OK')"
```

### Bước 2: Thu thập dữ liệu

Nếu bạn muốn train model với dữ liệu riêng:

```bash
python collect_data.py
```

Hướng dẫn:
1. Nhập tên class (ví dụ: A, B, C, Hello, Thanks, ...)
2. Đặt tay vào khung màu vàng
3. Nhấn SPACE để chụp ảnh (khuyến nghị 100-200 ảnh/class)
4. Nhấn N để chuyển sang class mới
5. Lặp lại cho tất cả các class cần nhận diện

Lưu ý:
- Chụp ở nhiều góc độ khác nhau
- Thay đổi ánh sáng
- Thay đổi background
- Thay đổi khoảng cách tay đến camera

### Bước 3: Chuẩn bị dữ liệu

Cấu trúc thư mục sau khi thu thập:

```
dataset/
├── train/
│   ├── A/
│   │   ├── A_0001.jpg
│   │   ├── A_0002.jpg
│   │   └── ...
│   ├── B/
│   ├── C/
│   └── ...
└── val/  (tùy chọn, nếu không có sẽ tự động split)
    ├── A/
    ├── B/
    └── ...
```

Nếu chỉ có thư mục train, YOLOv8 sẽ tự động split 80/20 cho train/val.

### Bước 4: Train model

```bash
python train_model.py
```

Quá trình training:
- Model sẽ train trong vài giờ (tùy thuộc vào số lượng dữ liệu và GPU)
- Kết quả lưu tại: `runs/classify/train/weights/best.pt`
- Copy file `best.pt` về thư mục gốc

Tùy chỉnh training trong `train_model.py`:
```python
MODEL_SIZE = "yolov8n-cls.pt"  # n=nhanh, s=cân bằng, m/l/x=chính xác
EPOCHS = 100                    # Số epoch
BATCH_SIZE = 32                 # Batch size (giảm nếu hết RAM)
```

### Bước 5: Test model

Test trên ảnh đơn:
```bash
python test_model.py --image test.jpg --save
```

Test trên thư mục:
```bash
python test_model.py --folder test_images/ --save
```

### Bước 6: Chạy ứng dụng real-time

```bash
python webcam_digits.py
```

Phím tắt:
- ESC: Thoát
- SPACE: Chụp và lưu ảnh
- H: Bật/tắt khung hướng dẫn

## Tùy chỉnh nâng cao

### 1. Thay đổi ngưỡng confidence

Mở `webcam_digits.py`, tìm dòng:
```python
CONFIDENCE_THRESHOLD = 0.5
```

Giảm xuống 0.3 nếu muốn nhạy hơn, tăng lên 0.7 nếu muốn chính xác hơn.

### 2. Thay đổi số lượng predictions hiển thị

```python
TOP_K = 3  # Hiển thị top 3, có thể đổi thành 5
```

### 3. Thay đổi kích thước input

Trong `webcam_digits.py`:
```python
results = model.predict(source=frame, imgsz=224, verbose=False)[0]
```

Đổi `imgsz=224` thành `imgsz=320` hoặc `imgsz=416` để tăng độ chính xác (nhưng chậm hơn).

### 4. Sử dụng GPU

YOLOv8 tự động dùng GPU nếu có CUDA. Kiểm tra:
```python
import torch
print(torch.cuda.is_available())  # True = có GPU
```

### 5. Export model sang định dạng khác

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Export sang ONNX (chạy nhanh hơn)
model.export(format='onnx')

# Export sang TensorFlow Lite (cho mobile)
model.export(format='tflite')

# Export sang CoreML (cho iOS)
model.export(format='coreml')
```

## Xử lý lỗi thường gặp

### Lỗi 1: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Lỗi 2: "Không mở được webcam"
- Kiểm tra webcam có hoạt động không
- Thử đổi `CAMERA_INDEX = 0` thành `CAMERA_INDEX = 1` trong `config.py`
- Kiểm tra quyền truy cập camera trong Windows Settings

### Lỗi 3: "CUDA out of memory" khi training
Giảm batch size trong `train_model.py`:
```python
BATCH_SIZE = 16  # hoặc 8
```

### Lỗi 4: Confidence luôn thấp
- Cải thiện chất lượng dữ liệu training
- Tăng số lượng ảnh training (>200 ảnh/class)
- Train thêm epochs
- Sử dụng model lớn hơn (yolov8s hoặc yolov8m)

### Lỗi 5: Model nhận diện sai
- Kiểm tra dữ liệu training có đúng không
- Đảm bảo các class không quá giống nhau
- Tăng độ đa dạng của dữ liệu (góc độ, ánh sáng, background)

## Tips để cải thiện độ chính xác

1. **Dữ liệu chất lượng cao**
   - Ít nhất 100-200 ảnh/class
   - Đa dạng góc độ, ánh sáng, background
   - Ảnh rõ nét, không bị mờ

2. **Data augmentation**
   YOLOv8 tự động áp dụng augmentation, nhưng có thể tùy chỉnh:
   ```python
   model.train(
       data=DATASET_PATH,
       augment=True,
       hsv_h=0.015,  # Hue augmentation
       hsv_s=0.7,    # Saturation
       hsv_v=0.4,    # Value
       degrees=10,   # Rotation
       translate=0.1, # Translation
       scale=0.5,    # Scaling
       flipud=0.0,   # Flip up-down
       fliplr=0.5,   # Flip left-right
   )
   ```

3. **Transfer learning**
   Sử dụng pretrained model (đã làm mặc định):
   ```python
   model = YOLO('yolov8n-cls.pt')  # Pretrained trên ImageNet
   ```

4. **Ensemble models**
   Kết hợp nhiều model để tăng độ chính xác:
   ```python
   model1 = YOLO('best_v1.pt')
   model2 = YOLO('best_v2.pt')
   
   # Lấy trung bình predictions
   pred1 = model1.predict(frame)[0].probs.data
   pred2 = model2.predict(frame)[0].probs.data
   avg_pred = (pred1 + pred2) / 2
   ```

## Tích hợp vào ứng dụng khác

### Flask Web App

```python
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame)[0]
        # Vẽ kết quả lên frame
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
```

### Tkinter Desktop App

```python
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.model = YOLO('best.pt')
        self.cap = cv2.VideoCapture(0)
        
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        self.label = tk.Label(window, text="", font=("Arial", 20))
        self.label.pack()
        
        self.update()
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model.predict(frame)[0]
            top_class = results.names[int(results.probs.top1)]
            
            self.label.config(text=f"Prediction: {top_class}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.photo = photo
        
        self.window.after(10, self.update)

root = tk.Tk()
app = SignLanguageApp(root)
root.mainloop()
```

## Tài nguyên tham khảo

- YOLOv8 Documentation: https://docs.ultralytics.com/
- OpenCV Documentation: https://docs.opencv.org/
- Sign Language MNIST Dataset: https://www.kaggle.com/datamunge/sign-language-mnist
- Repo gốc: https://github.com/kumarvivek9088/SignLanguageDetectionUsingCNN

## Liên hệ & Đóng góp

Nếu gặp vấn đề hoặc có đề xuất cải thiện, vui lòng tạo issue hoặc pull request.
