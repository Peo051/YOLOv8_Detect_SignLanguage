"""
Script test model trên ảnh tĩnh hoặc thư mục ảnh

Cách sử dụng:
1. Test một ảnh: python test_model.py --image path/to/image.jpg
2. Test thư mục: python test_model.py --folder path/to/folder
"""

from ultralytics import YOLO
import cv2
import os
import argparse
import numpy as np

# ==================== PARSE ARGUMENTS ====================
parser = argparse.ArgumentParser(description='Test model nhận diện ngôn ngữ ký hiệu')
parser.add_argument('--model', type=str, default='best.pt', help='Đường dẫn đến model')
parser.add_argument('--image', type=str, help='Đường dẫn đến ảnh test')
parser.add_argument('--folder', type=str, help='Đường dẫn đến thư mục ảnh test')
parser.add_argument('--top-k', type=int, default=3, help='Số lượng dự đoán hiển thị')
parser.add_argument('--save', action='store_true', help='Lưu kết quả')
args = parser.parse_args()

# ==================== LOAD MODEL ====================
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Không tìm thấy model: {args.model}")

print(f"Đang tải model: {args.model}")
model = YOLO(args.model)
print(f"✓ Đã tải model với {len(model.names)} classes")
print(f"Classes: {list(model.names.values())}\n")

# ==================== HÀM TEST ====================
def test_image(image_path, save_result=False):
    """Test một ảnh"""
    if not os.path.exists(image_path):
        print(f"⚠ Không tìm thấy: {image_path}")
        return None
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠ Không đọc được: {image_path}")
        return None
    
    # Dự đoán
    results = model.predict(source=img, imgsz=224, verbose=False)[0]
    probs = results.probs.data.cpu().numpy()
    
    # Lấy top-k predictions
    top_indices = np.argsort(probs)[::-1][:args.top_k]
    
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    predictions = []
    for i, idx in enumerate(top_indices):
        name = model.names[idx]
        conf = float(probs[idx])
        predictions.append((name, conf))
        
        status = "✓" if i == 0 else " "
        print(f"{status} {i+1}. {name:15s} {conf:6.2%} {'█' * int(conf * 50)}")
    
    # Vẽ kết quả lên ảnh
    if save_result:
        h, w = img.shape[:2]
        
        # Vẽ background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Vẽ predictions
        y_offset = 30
        for i, (name, conf) in enumerate(predictions):
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            text = f"{i+1}. {name}: {conf:.2%}"
            cv2.putText(img, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Progress bar
            bar_width = int(conf * 300)
            cv2.rectangle(img, (10, y_offset + 5), 
                          (10 + bar_width, y_offset + 15), color, -1)
            
            y_offset += 45
        
        # Lưu
        output_path = image_path.replace('.', '_result.')
        cv2.imwrite(output_path, img)
        print(f"\n✓ Đã lưu kết quả: {output_path}")
    
    return predictions

def test_folder(folder_path, save_result=False):
    """Test tất cả ảnh trong thư mục"""
    if not os.path.exists(folder_path):
        print(f"⚠ Không tìm thấy thư mục: {folder_path}")
        return
    
    # Lấy danh sách ảnh
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"⚠ Không tìm thấy ảnh trong: {folder_path}")
        return
    
    print(f"Tìm thấy {len(images)} ảnh\n")
    
    # Test từng ảnh
    results = []
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        preds = test_image(img_path, save_result)
        if preds:
            results.append((img_name, preds[0]))  # Lưu top-1
    
    # Thống kê
    print(f"\n{'='*60}")
    print(f"TỔNG KẾT")
    print(f"{'='*60}")
    
    # Đếm số lượng mỗi class
    class_counts = {}
    for img_name, (pred_class, conf) in results:
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
    
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {count} ảnh ({count/len(results)*100:.1f}%)")
    
    print(f"\nTổng: {len(results)} ảnh")

# ==================== MAIN ====================
if args.image:
    test_image(args.image, save_result=args.save)
elif args.folder:
    test_folder(args.folder, save_result=args.save)
else:
    print("⚠ Vui lòng chỉ định --image hoặc --folder")
    print("\nVí dụ:")
    print("  python test_model.py --image test.jpg")
    print("  python test_model.py --folder test_images/")
    print("  python test_model.py --image test.jpg --save")
