@echo off
echo.
echo ========================================
echo    Episode Video Generator
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo Checking dependencies...

REM Install requirements if needed
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating requirements...
pip install -r requirements_video.txt

echo.
echo ========================================
echo Starting Video Generator...
echo ========================================
echo.
echo The web interface will open at:
echo http://127.0.0.1:7861
echo.
echo Press Ctrl+C to stop the server
echo.

python video_generator.py

pause