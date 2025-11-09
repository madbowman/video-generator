@echo off
echo ========================================
echo Chatterbox TTS Installation
echo ========================================
echo.

echo Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found! Please install Python 3.10 or higher
    pause
    exit /b 1
)

echo.
echo Installing Chatterbox TTS...
echo This may take a few minutes...
echo.

pip install chatterbox-tts
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Installation failed. Please check your internet connection.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: python run_chatterbox_test.py
echo 2. Check CHATTERBOX_GUIDE.md for detailed usage
echo.
echo Your GPU: NVIDIA GeForce RTX 5070 (8GB VRAM)
echo Status: Perfect for Chatterbox! ✓
echo.
pause
