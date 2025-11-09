@echo off
echo ========================================
echo 🎙️  Chatterbox TTS Voice Generator
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Python not found! Please install Python 3.8 or higher
    echo.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

echo Checking required packages...
python -c "import gradio, torch, torchaudio" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Installing required packages...
    echo This may take a few minutes...
    pip install gradio torch torchaudio
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Package installation failed
        pause
        exit /b 1
    )
)

echo Checking Chatterbox TTS...
python -c "from chatterbox.tts import ChatterboxTTS" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Installing Chatterbox TTS...
    pip install chatterbox-tts
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Chatterbox TTS installation failed
        pause
        exit /b 1
    )
)

echo.
echo ✅ All dependencies ready!
echo.
echo 🚀 Starting Chatterbox TTS GUI...
echo    Opening in your default browser...
echo    Close this window to stop the server
echo.

python chatterbox_gui.py

echo.
echo 👋 Chatterbox TTS has been closed
pause