# Animation Batch Generator

Generate Pixar-style animation frames from episode scripts using AI.

## Quick Start

1. **Install Requirements**
   ```bash
   pip install gradio pillow requests
   ```

2. **Start ComfyUI**
   - Make sure ComfyUI is running on `http://127.0.0.1:8000`
   - Load the Disney Pixar Cartoon model (or Flux.1-dev)

3. **Run the Application**
   ```bash
   python animation_batch_generator.py
   ```
   
4. **Open your browser** to `http://127.0.0.1:7860`

## Features

### 📋 Timeline Tab
- Import episode scripts (.txt format)
- View all scenes in a timeline
- Edit individual scene prompts
- Adjust scene duration

### 🎨 Generate Tab
- Select AI model (Disney Pixar recommended)
- Generate all scenes with one click
- View progress in real-time
- Browse generated frames in gallery

### 👥 Characters Tab
- View all characters from episode
- See character descriptions and references

### ⚙️ Settings Tab
- Configure ComfyUI server URL
- Test connection
- View output directory

## Episode File Format

Your episode script should use this format:

```
IMAGE 1: Opening scene
A gnome named Bramble stands in a magical forest, morning light

IMAGE 2: Discovery
Bramble discovers a glowing mushroom, curious expression

IMAGE 3: Adventure begins
Bramble sets off on a path through the woods, determined
```

**Key points:**
- Each scene starts with `IMAGE N:` followed by scene title
- Prompt goes on the next line(s)
- The system automatically adds "pixar animation style" prefix
- Images are saved as 001.png, 002.png, etc. for easy video editing

## Output

Generated images are saved in the `outputs` folder with sequential numbering:
- `001.png` - First scene
- `002.png` - Second scene
- `003.png` - Third scene
- etc.

Import these directly into CapCut or your video editor!

## Troubleshooting

**Connection Failed**
- Make sure ComfyUI is running
- Check the URL in Settings tab
- Default: `http://127.0.0.1:8000`

**No Models Available**
- ComfyUI needs at least one model loaded
- Download Flux.1-dev or Disney Pixar Cartoon model
- Place in ComfyUI's models/checkpoints folder

**Import Failed**
- Check file format matches example above
- Each scene must start with "IMAGE N:"
- File should be plain text (.txt)

## Tips

1. **Character Consistency**: Mention character names and key features in each prompt
2. **Style Prefix**: Don't add "pixar animation style" - it's automatic
3. **Sequential Numbers**: Files are auto-numbered for video editing workflow
4. **Batch Size**: Test with 2-3 scenes first, then scale up
5. **Model Choice**: Disney Pixar model gives best results for this style

## System Requirements

- Python 3.8+
- ComfyUI installed and running
- At least one AI model (Flux.1-dev or Disney Pixar Cartoon)
- 8GB+ RAM recommended

---

**Note**: This tool generates images in the style of Pixar animation using AI models. It does not use or require actual Pixar software.
