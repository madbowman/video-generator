# Setup Chatterbox TTS on a New Computer

## Quick Setup (3 Steps)

### 1. Install Python 3.10
- Download from https://www.python.org/downloads/
- **Important**: Check "Add Python to PATH" during installation

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure for Your CPU
Edit `.env` file and set:
```
CPU_THREADS=<your CPU thread count>
```
To find your thread count:
- Windows: Open Task Manager → Performance → CPU → Logical processors
- Or run: `python -c "import psutil; print(psutil.cpu_count())"`

### 4. Run the Program
```bash
python chatterbox_gui.py
```
Then open: http://127.0.0.1:7862

## What Gets Configured via .env

✅ **CPU optimization** - Thread counts for maximum speed
✅ **Voice settings** - Exaggeration, temperature, pauses
✅ **Output paths** - Where audio files are saved
✅ **Device selection** - CPU or CUDA (when available)

## Files to Copy to New Computer

```
chatterbox_gui.py          (Main GUI)
chatterbox_story_tts.py    (TTS engine)
cpu_optimizer.py           (CPU optimization)
.env                       (Your settings - EDIT CPU_THREADS!)
requirements.txt           (Python dependencies)
start.bat                  (Quick launcher for Windows)
voices/                    (Your voice samples - optional)
```

## First Run on New Computer

1. Copy all files
2. Edit `.env` → Set `CPU_THREADS` to your CPU's thread count
3. Run: `pip install -r requirements.txt`
4. Run: `python chatterbox_gui.py` or double-click `start.bat`

**That's it!** The program will work exactly the same on the new computer.
