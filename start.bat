@echo off
echo.
echo ========================================
echo    Video Generator - Quick Start
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing/checking dependencies...
pip install -r requirements_video.txt >nul 2>&1

echo.
echo Starting Video Generator...
echo Web interface will open at: http://127.0.0.1:7861
echo.
echo Press Ctrl+C to stop
echo.

python video_generator.py

pause