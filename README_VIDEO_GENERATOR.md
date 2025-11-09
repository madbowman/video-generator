# 🎬 Episode Video Generator

A unified system that combines TTS audio generation and AI image creation to produce MP4 videos from episode scripts. Perfect for creating animated stories, audiobooks with visuals, and narrative content.

## ✨ Features

- **📝 Script Parsing**: Automatically extracts scenes from episode scripts
- **🎨 AI Image Generation**: Creates Pixar-style visuals using ComfyUI
- **🎙️ Text-to-Speech**: Generates natural voice narration for each scene
- **🎬 Video Creation**: Combines audio and images into MP4 videos
- **👥 Character Consistency**: Maintains character appearance across scenes
- **🌐 Web Interface**: Easy-to-use Gradio interface
- **⚙️ Flexible Settings**: Customizable models, durations, and output formats

## 🚀 Quick Start

### Prerequisites

1. **ComfyUI**: Must be running on `http://127.0.0.1:8000`
   - Install and set up ComfyUI
   - Load a Pixar/cartoon style model (e.g., Disney Pixar Cartoon)

2. **FFmpeg**: Required for video creation
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add to your system PATH

### Installation

1. **Easy Start** (Recommended):
   ```bash
   # Just double-click this file:
   start_video_generator.bat
   ```

2. **Manual Installation**:
   ```bash
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate

   # Install dependencies
   pip install -r requirements_video.txt

   # Start the generator
   python video_generator.py
   ```

3. **Open your browser** to `http://127.0.0.1:7861`

## 📁 Input File Format

### Episode Script Format
Your script file should contain scenes marked like this:
```
[SCENE 1: DAWN IN THE FOREST]
The first rays of morning light filtered through the ancient canopy...

[SCENE 2: VILLAGE AWAKENING]  
Roothollow was a village of perhaps one hundred gnomes...
```

### Image Prompts Format
Your prompts file should contain character descriptions and scene prompts:
```
CHARACTER REFERENCE (Use for consistency):

BRAMBLE QUICKWHISKER THISTLEWOOD (Protagonist):
- Age 40 (young adult gnome), height 2'10"
- Lean, wiry, athletic build
- PIXAR STYLE: Rounded features, expressive face...

SCENE 1: DAWN FOREST
Pixar animation style, morning forest scene with golden sunlight...

SCENE 2: GNOME VILLAGE  
Pixar style gnome village built into trees and hillsides...
```

## 🎯 Usage Workflow

1. **Upload Files**: 
   - Episode script (.txt)
   - Image prompts (.txt)
   - Voice sample (.wav/.mp3) - Optional

2. **Configure Settings**:
   - Select AI model from ComfyUI
   - Set output video name

3. **Generate Video**:
   - Click "Generate Video"
   - Watch progress as scenes are processed
   - Download your MP4 when complete

## ⚙️ System Requirements

### Minimum Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for temp files
- **Python**: 3.8 or higher

### For GPU Acceleration (Optional)
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: Compatible CUDA installation
- **Note**: System works on CPU-only, just slower

## 🔧 Configuration

### ComfyUI Setup
1. Install ComfyUI and start the server
2. Load appropriate models:
   - **Recommended**: Disney Pixar Cartoon v1.0
   - **Alternative**: Flux.1-dev, SDXL-based cartoon models
3. Ensure server runs on `http://127.0.0.1:8000`

### Audio Configuration
- Uses Chatterbox TTS for natural voice generation
- Supports voice cloning with sample files
- Automatic CPU optimization for performance

### Video Output Settings
- **Format**: MP4 (H.264/AAC)
- **Resolution**: 1024x1024 (adjustable)
- **FPS**: 24 (adjustable)
- **Duration**: Auto-detected from audio or 5s default per scene

## 📝 File Structure

```
auto video/
├── video_generator.py          # Main unified application
├── requirements_video.txt      # Dependencies
├── start_video_generator.bat   # Easy launcher
├── video_outputs/             # Generated videos
├── Episode_01_Script_*.txt    # Example script
├── Episode_01_Image_*.txt     # Example prompts
├── audio/                     # TTS system
│   ├── chatterbox_gui.py
│   ├── chatterbox_story_tts.py
│   └── requirements.txt
└── image/                     # Image generation system
    ├── app_final_timeline.py
    └── requirements.txt
```

## 🎨 Supported Models

### Image Generation (ComfyUI)
- Disney Pixar Cartoon v1.0 (Recommended)
- Flux.1-dev
- SDXL-based cartoon/animation models
- Custom trained character models

### Audio Generation
- Chatterbox TTS with voice cloning
- Multi-language support (23+ languages)
- Emotion and style control

## 🚨 Troubleshooting

### Common Issues

**"ComfyUI connection failed"**
- Ensure ComfyUI is running on port 8000
- Check firewall settings
- Verify model is loaded in ComfyUI

**"FFmpeg not found"**
- Install FFmpeg from official website
- Add FFmpeg to system PATH
- Restart command prompt/application

**"Audio generation failed"**
- Check if chatterbox-tts is installed
- Verify voice sample file format
- Try CPU-only mode if GPU issues

**"Out of memory"**
- Reduce image resolution in code
- Use CPU instead of GPU
- Process fewer scenes at once

### Performance Optimization

1. **For faster generation**:
   - Use GPU acceleration when available
   - Reduce image resolution (edit `width`/`height` in code)
   - Use simpler AI models

2. **For better quality**:
   - Use higher resolution settings
   - Select premium AI models
   - Provide high-quality voice samples

## 🤝 Integration with Existing Systems

This unified system combines:
- **Audio folder**: Chatterbox TTS functionality
- **Image folder**: ComfyUI image generation
- **New features**: Video creation, scene parsing, workflow automation

The existing audio and image systems remain functional independently.

## 📋 Example Episode Files

Included in the workspace:
- `Episode_01_Script_The_Wanderer_EXPANDED.txt`
- `Episode_01_Image_Prompts_The_Wanderer_EXPANDED.txt`

These demonstrate the proper format for input files.

## 🔄 Workflow Details

1. **Parse** → Extract scenes and characters from input files
2. **Generate Audio** → Create TTS narration for each scene  
3. **Generate Images** → Create AI visuals for each scene
4. **Combine** → Use FFmpeg to create final MP4 video
5. **Output** → Save to video_outputs/ directory

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all prerequisites are installed
3. Test ComfyUI connection separately
4. Check console output for specific error messages