@echo off
chcp 65001 >nul
echo ========================================
echo   THIẾT LẬP PROJECT
echo ========================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Không tìm thấy Python
    echo Vui lòng cài đặt Python 3.8+ từ https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Cài đặt dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Cài đặt thất bại
    pause
    exit /b 1
)

echo.
echo [2/3] Tạo thư mục...
if not exist "captured_signs" mkdir captured_signs
if not exist "dataset\train" mkdir dataset\train
if not exist "dataset\val" mkdir dataset\val

echo.
echo [3/3] Kiểm tra hệ thống...
python check_system.py

echo.
echo ========================================
echo   HOÀN TẤT THIẾT LẬP
echo ========================================
echo.
echo Chạy ứng dụng:
echo   run.bat
echo   hoặc: python webcam_digits.py
echo.
pause
