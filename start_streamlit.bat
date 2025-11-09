@echo off
echo.
echo ========================================
echo    Video Generator - Streamlit Version
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
pip install streamlit requests pillow torch numpy scipy >nul 2>&1

echo.
echo Starting Streamlit Video Generator...
echo Web interface will open automatically at: http://localhost:8501
echo.
echo Press Ctrl+C to stop
echo.

streamlit run app_video_streamlit.py

pause