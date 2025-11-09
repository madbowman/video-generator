@echo off
echo.
echo 🎨 Animation Batch Generator - Quick Start
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo 🚀 Starting Animation Batch Generator...
echo.
echo The application will open in your browser at:
echo    👉 http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the application
streamlit run app_streamlit.py
pause

REM Check if ComfyUI is running
echo 🔍 Checking ComfyUI connection...
curl -s http://127.0.0.1:8000/system_stats >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Warning: Could not connect to ComfyUI on port 8000
    echo    Make sure ComfyUI is running before generating images.
    echo    Start ComfyUI with: python main.py --port 8000
) else (
    echo ✅ ComfyUI is running on port 8000
)

echo.
echo 🚀 Starting Pixar Batch Generator...
echo    The web interface will open at http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the application
python pixar_batch_generator.py

pause
