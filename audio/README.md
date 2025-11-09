# 🎙️ Chatterbox TTS Voice Generator

A powerful, local text-to-speech system with advanced emotion control, multi-speaker dialog, and voice cloning capabilities. Perfect for creating audiobooks, podcasts, and story narration.

## ✨ Features

- **🎭 Emotion Control**: Per-sentence emotion tags for dramatic storytelling
- **👥 Multi-Speaker Dialog**: Up to 4 different voices with flexible character naming
- **🗣️ Voice Cloning**: Clone any voice from 10-20 seconds of audio
- **🌍 Multilingual**: 23 languages supported
- **🎵 Voice Conversion**: Transform existing audio to different voices
- **� Progress Tracking**: Real-time progress bars with time estimation
- **�💻 100% Local**: No internet required, complete privacy
- **🖥️ CPU Optimized**: Optimized for CPU performance with automatic threading (uses 4-8 cores)
- **🚀 Easy Launch**: One-click start.bat launcher

## 🚀 Quick Start

### Easy Launch
**Double-click `start.bat`** - handles all dependencies automatically!

### Manual Installation
```bash
# Install required dependencies
pip install chatterbox-tts gradio

# Launch the GUI
python chatterbox_gui.py
```

Open your browser to **http://localhost:7860**

### Basic Usage
```python
from chatterbox_story_tts import StoryNarrator

# Initialize narrator (CPU-optimized by default)
narrator = StoryNarrator(device="cpu")

# Generate speech
narrator.narrate_story(
    story_text="Your story text here...",
    output_path="output.wav"
)
```

## 🎭 GUI Features

### Tab 1: Basic TTS
Simple text-to-speech generation that **ignores all [tags]**
- Upload voice sample for cloning
- Adjust emotion and speed sliders
- Perfect for clean text without any formatting

### Tab 2: Emotion Tags
Control emotions using **named emotion tags**:
```
[happy]Hello there! How are you today?
[sad]I'm feeling a bit down lately.
[angry]This is completely unacceptable!
[excited]I can't wait for the party!
```

**Available emotions:** `happy`, `sad`, `angry`, `excited`, `tired`, `scared`, `calm`, `dramatic`, `whisper`, `shouting`, `confused`, `surprised`, `concerned`

### Tab 3: Multi-Speaker Dialog
Create conversations with **Speaker 1-4** system:

**Setup:**
1. Upload voice files to Speaker 1-4 slots
2. Enter character names (e.g., "narrator", "hero", "villain")
3. Use those names in your dialog

```
[narrator]The story begins in a quiet village.
[hero,happy]"What a beautiful day!"
[villain,angry]"Not for long!"
[narrator]The confrontation begins.
```

### Tab 4: Voice Conversion
Transform existing audio to different voices
- Upload source audio + target voice sample
- Keeps words, changes voice

### Tab 5: Multilingual
Generate speech in 23 languages with voice cloning support

## ⚙️ Parameters Guide

### Exaggeration (0.25-2.0)
- `0.25-0.4`: Calm, monotone
- `0.5`: Natural conversation  
- `0.7-0.9`: Expressive storytelling
- `1.0-2.0`: Dramatic, theatrical

### CFG Weight (0.2-1.0) 
- `0.2-0.3`: Very slow, deliberate
- `0.35-0.4`: Audiobook pace
- `0.5`: Normal speed
- `0.6-0.8`: Fast, energetic

### Temperature (0.0-1.0)
- `0.0-0.3`: Consistent, predictable
- `0.5-0.7`: Natural variation
- `0.8-1.0`: More creative

## 📁 Project Structure

### Core Files
- `chatterbox_gui.py` - Main GUI application
- `chatterbox_story_tts.py` - Core TTS functionality
- `run_chatterbox_test.py` - Test script and examples

### Utilities
- `check_gpu.py` - GPU detection and verification
- `install_chatterbox.bat` - Windows installation script

### Configuration
- `.env` - Environment variables
- `env.template.txt` - Environment template
- `example_story.txt` - Example story content

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Use CPU instead
narrator = StoryNarrator(device="cpu")

# Or clear cache between generations
torch.cuda.empty_cache()
```

### Audio Quality Issues
- **Robotic voice**: Increase exaggeration to 0.7-0.9
- **Too fast**: Increase cfg_weight to 0.6-0.7
- **Too slow**: Decrease cfg_weight to 0.3-0.4
- **Poor voice match**: Use 15-20 seconds of clean reference audio

### Import Errors
```bash
pip install chatterbox-tts --user
```

### CPU Performance Issues
**Slow generation**: 
- Run `python cpu_optimizer.py` to check system optimization
- Close other applications to free up CPU/RAM
- Use shorter text segments for testing

**Memory errors**:
- Reduce story length or split into chapters
- Restart Python session between long generations
- Monitor RAM usage during processing

### Dimension Mismatch (Fixed)
The project includes fixes for tensor dimension mismatches that were common in earlier versions.

## 💡 Tips for Best Results

1. **Voice Cloning**: Use 15-20 seconds of clean, clear audio
2. **Long Stories**: Use sentence-by-sentence processing with emotion tags
3. **Multi-Speaker**: Ensure each speaker has a distinct voice sample
4. **Dramatic Reading**: Use low cfg_weight (0.25-0.35) with high exaggeration (1.0-1.5)
5. **Natural Conversation**: Use moderate settings (cfg_weight: 0.4-0.5, exaggeration: 0.5-0.7)

## 🎯 Use Cases

- **Audiobook Production**: Long-form narration with emotion control
- **Podcast Creation**: Multi-speaker conversations
- **Character Voices**: Voice conversion for consistent character portrayals
- **Language Learning**: Multilingual content generation
- **Accessibility**: Text-to-speech for visual impairments

## 📊 System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB+ system RAM (16GB+ for long stories)
- **Python**: 3.8+ with PyTorch CPU support
- **Storage**: 2GB+ free space for models

### CPU Performance Tips
- **4+ cores**: Optimal performance with automatic threading (will use 4-8 cores)
- **8+ GB RAM**: Smooth processing for long stories
- **16+ GB RAM**: Best performance for complex multi-speaker content
- **SSD Storage**: Faster model loading and audio processing

### Performance Improvements
- **Real-time progress tracking** with accurate ETA calculation
- **Multi-core CPU utilization** (automatically uses 4-8 cores)
- **Memory optimization** with automatic cleanup
- **Batch processing** for improved efficiency

## 🤝 Contributing

This is a local voice generation project. Feel free to modify and enhance the code for your specific needs.

## 📄 License

Check individual component licenses. Chatterbox TTS follows its own licensing terms.

---

**Note**: This project prioritizes privacy and local processing. All voice generation happens on your machine without external API calls or data transmission.