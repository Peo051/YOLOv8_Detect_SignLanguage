"""
File cấu hình chung cho project nhận diện ngôn ngữ ký hiệu
"""

# ==================== MODEL ====================
MODEL_PATH = "best.pt"
MODEL_INPUT_SIZE = 224

# ==================== DETECTION ====================
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy tối thiểu
TOP_K_PREDICTIONS = 3       # Số lượng dự đoán hiển thị

# ==================== CAMERA ====================
CAMERA_INDEX = 0            # 0 = camera mặc định, 1 = camera thứ 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ==================== UI ====================
SHOW_FPS = True
SHOW_GUIDE_BOX = True
GUIDE_BOX_SIZE = 250

# Colors (BGR format)
COLOR_PRIMARY = (0, 255, 0)      # Xanh lá
COLOR_SECONDARY = (200, 200, 200) # Xám
COLOR_WARNING = (0, 0, 255)       # Đỏ
COLOR_INFO = (255, 255, 0)        # Vàng
COLOR_TEXT = (255, 255, 255)      # Trắng

# ==================== DATA COLLECTION ====================
DATASET_PATH = "dataset"
IMAGES_PER_CLASS = 100

# ==================== TRAINING ====================
PRETRAINED_MODEL = "yolov8n-cls.pt"  # n, s, m, l, x
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 20  # Early stopping

# Device sẽ tự động phát hiện (GPU hoặc CPU)
# Không cần cấu hình thủ công

# ==================== EXPORT ====================
SAVE_DIR = "captured_signs"
EXPORT_FORMAT = "jpg"  # jpg, png

# ==================== LOGGING ====================
LOG_PREDICTIONS = False
LOG_FILE = "predictions.log"
