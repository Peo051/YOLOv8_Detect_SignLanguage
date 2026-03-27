@echo off
chcp 65001 >nul
echo ========================================
echo   NHẬN DIỆN NGÔN NGỮ KÝ HIỆU
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

REM Kiểm tra best.pt
if not exist "best.pt" (
    echo [!] Không tìm thấy best.pt
    echo.
    echo Bạn muốn tải xuống từ Google Drive không? (Y/N)
    set /p download="Nhập lựa chọn: "
    if /i "%download%"=="Y" (
        echo.
        echo Đang tải xuống...
        python download_from_drive.py
        if errorlevel 1 (
            echo.
            echo [X] Tải xuống thất bại
            echo Vui lòng tải thủ công từ:
            echo https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Vui lòng tải best.pt từ:
        echo https://drive.google.com/drive/folders/1kCTsv1-qlBXgYP5aEdbZQNG3yXOqVbf5
        pause
        exit /b 1
    )
)

REM Chạy ứng dụng
echo.
echo [✓] Đang khởi động ứng dụng...
echo.
python webcam_digits.py

if errorlevel 1 (
    echo.
    echo [X] Có lỗi xảy ra
    echo Chạy lệnh sau để kiểm tra:
    echo   python check_system.py
    pause
)
