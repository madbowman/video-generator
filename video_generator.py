#!/usr/bin/env python3
"""
Unified Video Generator
Merges audio TTS and image generation to create MP4 videos from episode scripts
"""

import os
import re
import json
import time
import requests
import gradio as gr
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from scipy.io import wavfile
import subprocess
import tempfile
import io

# Import audio components
try:
    from audio.chatterbox_story_tts import StoryNarrator
    AUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ Audio module not found. Audio generation will be disabled.")
    AUDIO_AVAILABLE = False

class VideoGenerator:
    def __init__(self, comfyui_url="http://127.0.0.1:8000", output_dir="video_outputs"):
        self.comfyui_url = comfyui_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio setup
        self.narrator = None
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"🎮 GPU DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("🖥️ Using CPU")
            
        # Create temp directories
        self.temp_audio_dir = self.output_dir / "temp_audio"
        self.temp_image_dir = self.output_dir / "temp_images"
        self.temp_audio_dir.mkdir(exist_ok=True)
        self.temp_image_dir.mkdir(exist_ok=True)
        
    def parse_episode_files(self, script_file, prompts_file):
        """Parse both script and prompts files to extract scenes"""
        scenes = []
        characters = {}
        
        try:
            # Parse script file for narrative text
            print(f"Parsing script: {script_file}")
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
                
            # Parse prompts file for visual descriptions and characters
            print(f"Parsing prompts: {prompts_file}")
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts_content = f.read()
            
            # Extract characters from prompts file
            characters = self._extract_characters(prompts_content)
            
            # Extract scenes by matching script and prompts
            scenes = self._extract_scenes(script_content, prompts_content)
            
            print(f"Found {len(scenes)} scenes and {len(characters)} characters")
            return scenes, characters
            
        except Exception as e:
            print(f"Error parsing files: {e}")
            return [], {}
    
    def _extract_characters(self, prompts_content):
        """Extract character descriptions from prompts file"""
        characters = {}
        lines = prompts_content.split('\n')
        
        current_character = None
        collecting_description = False
        
        for line in lines:
            line = line.strip()
            
            # Look for character names (all caps with descriptors)
            if line.endswith(':') and '(' in line and line.isupper():
                char_name = line.split('(')[0].strip().rstrip(':')
                if len(char_name) > 2:
                    current_character = char_name
                    characters[current_character] = {'name': char_name, 'description': [line]}
                    collecting_description = True
                    continue
            
            # Collect character description lines
            if collecting_description and current_character:
                if line.startswith('-') or line.startswith('•'):
                    characters[current_character]['description'].append(line)
                elif line.startswith('PIXAR STYLE:'):
                    characters[current_character]['description'].append(line)
                elif not line or line.startswith('═') or ('SCENE' in line and line.isupper()):
                    collecting_description = False
                    current_character = None
        
        # Convert descriptions to single strings
        for char in characters:
            characters[char]['full_description'] = ' '.join(characters[char]['description'])
        
        return characters
    
    def _extract_scenes(self, script_content, prompts_content):
        """Extract scenes by matching script text with image prompts"""
        scenes = []
        
        # Extract scenes from script
        script_scenes = re.findall(r'\[SCENE \d+:([^\]]+)\](.*?)(?=\[SCENE|\Z)', script_content, re.DOTALL)
        
        # Extract image prompts from prompts file - look for IMAGE entries
        prompt_lines = prompts_content.split('\n')
        image_prompts = {}
        
        i = 0
        while i < len(prompt_lines):
            line = prompt_lines[i].strip()
            
            # Look for IMAGE entries like "IMAGE 1: DAWN FOREST"
            image_match = re.match(r'IMAGE (\d+):', line, re.IGNORECASE)
            if image_match:
                scene_num = int(image_match.group(1))
                image_prompts[scene_num] = {'title': line, 'prompts': []}
                i += 1
                
                # Collect the prompt text that follows
                while i < len(prompt_lines):
                    prompt_line = prompt_lines[i].strip()
                    if not prompt_line:  # Empty line
                        break
                    if prompt_line.startswith('IMAGE') or prompt_line.startswith('SCENE'):  # Next section
                        break
                    if prompt_line.startswith('═'):  # Separator
                        break
                    
                    # Add meaningful prompt lines
                    if len(prompt_line) > 10 and not prompt_line.startswith('CHARACTER'):
                        image_prompts[scene_num]['prompts'].append(prompt_line)
                    i += 1
                continue
            
            # Also look for SCENE entries as backup
            scene_match = re.match(r'SCENE (\d+):', line, re.IGNORECASE)
            if scene_match:
                scene_num = int(scene_match.group(1))
                if scene_num not in image_prompts:
                    image_prompts[scene_num] = {'title': line, 'prompts': []}
                i += 1
                
                # Collect prompts for this scene
                while i < len(prompt_lines):
                    prompt_line = prompt_lines[i].strip()
                    if not prompt_line or prompt_line.startswith('SCENE') or prompt_line.startswith('IMAGE'):
                        break
                    if len(prompt_line) > 10 and 'Pixar' in prompt_line:
                        image_prompts[scene_num]['prompts'].append(prompt_line)
                    i += 1
                continue
            
            i += 1
        
        # Combine script and prompts
        for i, (scene_title, scene_text) in enumerate(script_scenes, 1):
            prompts = image_prompts.get(i, {}).get('prompts', [])
            
            # If no specific prompts found, create a default one
            if not prompts:
                prompts = [f"Pixar animation style scene: {scene_title.strip()}"]
            
            scene_data = {
                'number': i,
                'title': scene_title.strip(),
                'script_text': scene_text.strip(),
                'image_prompts': prompts,
                'duration': 5.0  # Default 5 seconds per scene
            }
            scenes.append(scene_data)
        
        return scenes
    
    def generate_scene_audio(self, scene_text, scene_number, voice_file=None):
        """Generate TTS audio for a scene"""
        if not AUDIO_AVAILABLE:
            print("⚠️ Audio generation not available")
            return None
            
        try:
            # Initialize narrator if needed
            if self.narrator is None:
                self.narrator = StoryNarrator(
                    voice_sample_path=voice_file,
                    device=self.device
                )
            
            # Clean text for TTS (remove scene markers, etc.)
            clean_text = re.sub(r'\[.*?\]', '', scene_text)
            clean_text = clean_text.strip()
            
            if not clean_text:
                return None
            
            # Generate audio
            output_path = self.temp_audio_dir / f"scene_{scene_number:03d}.wav"
            print(f"Generating audio for scene {scene_number}: {clean_text[:50]}...")
            
            self.narrator.narrate_story(
                story_text=clean_text,
                output_path=str(output_path)
            )
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error generating audio for scene {scene_number}: {e}")
            return None
    
    def generate_scene_image(self, scene_prompts, characters, scene_number, model_name="disneyPixarCartoon_v10.safetensors", custom_seed=None):
        """Generate image for a scene using ComfyUI (matching image folder approach)"""
        try:
            # Combine scene prompts with character descriptions
            if not scene_prompts:
                base_prompt = "Pixar animation style scene"
            else:
                # Use the first available prompt
                base_prompt = scene_prompts[0] if isinstance(scene_prompts, list) else str(scene_prompts)
            
            # Add character descriptions to improve consistency
            character_context = ""
            for char_name, char_data in characters.items():
                if char_name.lower() in base_prompt.lower():
                    if isinstance(char_data, dict):
                        character_context += f" {char_data.get('full_description', '')}"
                    else:
                        character_context += f" {char_data}"
            
            full_prompt = f"{base_prompt} {character_context}".strip()
            
            # Use custom seed or generate one based on scene and time
            seed = custom_seed if custom_seed else int(time.time() * 1000) + scene_number
            
            # Create ComfyUI workflow
            workflow = self._create_comfyui_workflow(full_prompt, model_name, seed)
            
            print(f"Generating image for scene {scene_number}: {full_prompt[:50]}...")
            print(f"Using seed: {seed}")
            
            # Submit to ComfyUI and wait for result
            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            
            prompt_id = response.json()["prompt_id"]
            print(f"Submitted workflow, prompt ID: {prompt_id}")
            
            # Wait for completion with longer timeout
            max_wait = 180  # 3 minutes
            wait_time = 0
            
            while wait_time < max_wait:
                time.sleep(3)
                wait_time += 3
                
                try:
                    history_response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
                    if history_response.status_code == 200:
                        history = history_response.json()
                        if prompt_id in history and 'outputs' in history[prompt_id]:
                            # Found completed job, extract image
                            for node_output in history[prompt_id]['outputs'].values():
                                if 'images' in node_output:
                                    img_info = node_output['images'][0]
                                    
                                    # Download the image
                                    img_response = requests.get(f"{self.comfyui_url}/view", 
                                                               params={
                                                                   "filename": img_info['filename'], 
                                                                   "subfolder": img_info.get('subfolder', ''), 
                                                                   "type": "output"
                                                               })
                                    
                                    if img_response.status_code == 200:
                                        # Save image
                                        img = Image.open(io.BytesIO(img_response.content))
                                        output_path = self.temp_image_dir / f"scene_{scene_number:03d}.png"
                                        img.save(output_path)
                                        print(f"✅ Saved: {output_path}")
                                        return str(output_path)
                            break
                except Exception as e:
                    print(f"Error checking history: {e}")
                    continue
            
            print(f"❌ Timeout waiting for scene {scene_number} after {max_wait}s")
            return None
            
        except Exception as e:
            print(f"❌ Error generating image for scene {scene_number}: {e}")
            return None
    
    def _create_comfyui_workflow(self, prompt, model_name, seed, width=1024, height=1024):
        """Create ComfyUI workflow JSON"""
        return {
            "3": {"inputs": {"seed": seed, "steps": 20, "cfg": 7.0, "sampler_name": "euler", 
                            "scheduler": "normal", "denoise": 1.0, "model": ["4", 0], 
                            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["11", 0]}, 
                 "class_type": "KSampler"},
            "4": {"inputs": {"ckpt_name": model_name}, "class_type": "CheckpointLoaderSimple"},
            "6": {"inputs": {"text": prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"text": "low quality, blurry, distorted, ugly, bad anatomy, watermark", 
                            "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": f"scene_{seed}", "images": ["8", 0]}, 
                 "class_type": "SaveImage"},
            "11": {"inputs": {"width": width, "height": height, "batch_size": 1}, 
                  "class_type": "EmptyLatentImage"}
        }
    
    def _submit_to_comfyui(self, workflow, scene_number):
        """Submit workflow to ComfyUI and get result"""
        try:
            # Queue the workflow
            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            
            prompt_id = response.json()["prompt_id"]
            print(f"Submitted workflow, prompt ID: {prompt_id}")
            
            # Wait for completion and get result
            image_path = self._wait_for_completion(prompt_id, scene_number)
            return image_path
            
        except Exception as e:
            print(f"Error submitting to ComfyUI: {e}")
            return None
    
    def _wait_for_completion(self, prompt_id, scene_number, max_wait=120):
        """Wait for ComfyUI to complete and retrieve the image"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Check queue status
                response = requests.get(f"{self.comfyui_url}/queue")
                queue_data = response.json()
                
                # Check if our job is still in queue
                running = queue_data.get('queue_running', [])
                pending = queue_data.get('queue_pending', [])
                
                found_in_queue = any(item[1] == prompt_id for item in running + pending)
                
                if not found_in_queue:
                    # Job completed, get the image
                    return self._get_generated_image(prompt_id, scene_number)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"Error checking queue: {e}")
                time.sleep(2)
        
        print(f"Timeout waiting for scene {scene_number}")
        return None
    
    def _get_generated_image(self, prompt_id, scene_number):
        """Retrieve generated image from ComfyUI"""
        try:
            # Get history
            response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
            history = response.json()
            
            if prompt_id not in history:
                print(f"No history found for prompt {prompt_id}")
                return None
            
            # Find the output images
            outputs = history[prompt_id].get('outputs', {})
            
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    for image_info in node_output['images']:
                        filename = image_info['filename']
                        
                        # Download the image
                        img_response = requests.get(f"{self.comfyui_url}/view", 
                                                   params={'filename': filename, 'type': 'output'})
                        
                        if img_response.status_code == 200:
                            # Save image
                            output_path = self.temp_image_dir / f"scene_{scene_number:03d}.png"
                            with open(output_path, 'wb') as f:
                                f.write(img_response.content)
                            
                            print(f"Saved image: {output_path}")
                            return str(output_path)
            
            print(f"No images found in outputs for prompt {prompt_id}")
            return None
            
        except Exception as e:
            print(f"Error getting generated image: {e}")
            return None
    
    def create_video(self, scenes, output_filename, fps=24):
        """Combine audio and images into MP4 video using FFmpeg"""
        try:
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None, "❌ FFmpeg not found. Please install FFmpeg to create videos."
            
            # Create temporary files list for FFmpeg
            temp_dir = tempfile.mkdtemp()
            concat_file = os.path.join(temp_dir, 'concat.txt')
            
            video_segments = []
            
            for i, scene in enumerate(scenes):
                scene_num = scene['number']
                duration = scene.get('duration', 5.0)
                
                image_path = self.temp_image_dir / f"scene_{scene_num:03d}.png"
                audio_path = self.temp_audio_dir / f"scene_{scene_num:03d}.wav"
                segment_path = os.path.join(temp_dir, f"segment_{scene_num:03d}.mp4")
                
                # Check if files exist
                if not image_path.exists():
                    print(f"⚠️ Missing image for scene {scene_num}")
                    continue
                
                # Create video segment for this scene
                cmd = ['ffmpeg', '-y', '-loop', '1', '-i', str(image_path)]
                
                if audio_path.exists():
                    # Use audio duration if available
                    cmd.extend(['-i', str(audio_path)])
                    cmd.extend(['-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-pix_fmt', 'yuv420p'])
                else:
                    # Use specified duration
                    cmd.extend(['-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p'])
                
                cmd.extend(['-r', str(fps), segment_path])
                
                print(f"Creating segment {scene_num}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    video_segments.append(segment_path)
                    print(f"✅ Created segment {scene_num}")
                else:
                    print(f"❌ Error creating segment {scene_num}: {result.stderr}")
            
            if not video_segments:
                return None, "❌ No video segments were created successfully"
            
            # Create concat file for FFmpeg
            with open(concat_file, 'w') as f:
                for segment in video_segments:
                    f.write(f"file '{segment}'\n")
            
            # Combine all segments
            output_path = self.output_dir / f"{output_filename}.mp4"
            concat_cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                '-i', concat_file, '-c', 'copy', str(output_path)
            ]
            
            print(f"Combining segments: {' '.join(concat_cmd)}")
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Cleanup temp files
                for segment in video_segments:
                    try:
                        os.remove(segment)
                    except:
                        pass
                os.remove(concat_file)
                os.rmdir(temp_dir)
                
                return str(output_path), f"✅ Video created successfully: {output_path}"
            else:
                return None, f"❌ Error combining segments: {result.stderr}"
                
        except Exception as e:
            return None, f"❌ Error creating video: {str(e)}"
    
    def get_available_models(self):
        """Get available ComfyUI models"""
        try:
            response = requests.get(f"{self.comfyui_url}/object_info", timeout=5)
            if response.status_code == 200:
                object_info = response.json()
                if "CheckpointLoaderSimple" in object_info:
                    models = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                    return sorted(models)
        except:
            pass
        return ["disneyPixarCartoon_v10.safetensors", "flux1-dev.safetensors"]


def create_gradio_interface():
    """Create Gradio interface for video generation"""
    
    generator = VideoGenerator()
    
    # Shared state for scenes and characters
    scenes_state = gr.State([])
    characters_state = gr.State({})
    
    def parse_files(script_file, prompts_file):
        """Parse episode files and return scenes info"""
        try:
            if not script_file or not prompts_file:
                return "❌ Please upload both files", [], {}
            
            scenes, characters = generator.parse_episode_files(script_file.name, prompts_file.name)
            
            if not scenes:
                return "❌ No scenes found in files", [], {}
            
            scene_info = f"✅ Found {len(scenes)} scenes and {len(characters)} characters\n\n"
            scene_info += "Scenes:\n"
            for scene in scenes[:5]:  # Show first 5 scenes
                scene_info += f"• Scene {scene['number']}: {scene['title']}\n"
            if len(scenes) > 5:
                scene_info += f"... and {len(scenes) - 5} more scenes\n"
            
            scene_info += f"\nCharacters: {', '.join(characters.keys())}"
            
            return scene_info, scenes, characters
            
        except Exception as e:
            return f"❌ Error parsing files: {str(e)}", [], {}
    
    def generate_images_only(scenes, characters, model_name, progress=gr.Progress()):
        """Generate only images for all scenes"""
        try:
            if not scenes:
                return "❌ No scenes loaded. Please parse files first.", []
            
            progress(0, desc="Starting image generation...")
            
            generated_images = []
            total_scenes = len(scenes)
            
            for i, scene in enumerate(scenes):
                scene_progress = i / total_scenes
                progress(scene_progress, desc=f"Generating image for scene {scene['number']}: {scene['title']}")
                
                image_path = generator.generate_scene_image(
                    scene['image_prompts'], 
                    characters, 
                    scene['number'], 
                    model_name
                )
                
                if image_path:
                    generated_images.append(image_path)
                    scene['image_path'] = image_path
            
            progress(1.0, desc="Image generation complete!")
            
            if generated_images:
                return f"✅ Generated {len(generated_images)} images successfully!", generated_images
            else:
                return "❌ No images were generated", []
                
        except Exception as e:
            return f"❌ Error generating images: {str(e)}", []
    
    def generate_audio_only(scenes, voice_file, progress=gr.Progress()):
        """Generate only audio for all scenes"""
        try:
            if not scenes:
                return "❌ No scenes loaded. Please parse files first.", []
            
            if not AUDIO_AVAILABLE:
                return "❌ Audio system not available", []
            
            progress(0, desc="Starting audio generation...")
            
            generated_audio = []
            total_scenes = len(scenes)
            
            for i, scene in enumerate(scenes):
                scene_progress = i / total_scenes
                progress(scene_progress, desc=f"Generating audio for scene {scene['number']}: {scene['title']}")
                
                audio_path = generator.generate_scene_audio(
                    scene['script_text'], 
                    scene['number'], 
                    voice_file.name if voice_file else None
                )
                
                if audio_path:
                    generated_audio.append(audio_path)
                    scene['audio_path'] = audio_path
            
            progress(1.0, desc="Audio generation complete!")
            
            if generated_audio:
                return f"✅ Generated {len(generated_audio)} audio files successfully!", generated_audio
            else:
                return "❌ No audio files were generated", []
                
        except Exception as e:
            return f"❌ Error generating audio: {str(e)}", []
    
    def regenerate_single_image(scenes, characters, scene_number, model_name, custom_seed, progress=gr.Progress()):
        """Regenerate a single image with optional custom seed"""
        try:
            if not scenes:
                return "❌ No scenes loaded", []
            
            scene_number = int(scene_number)
            
            # Find the scene
            target_scene = None
            for scene in scenes:
                if scene['number'] == scene_number:
                    target_scene = scene
                    break
            
            if not target_scene:
                return f"❌ Scene {scene_number} not found", []
            
            progress(0.5, desc=f"Regenerating image for scene {scene_number}")
            
            # Use custom seed if provided, otherwise generate random one
            seed = int(custom_seed) if custom_seed else None
            
            image_path = generator.generate_scene_image(
                target_scene['image_prompts'], 
                characters, 
                scene_number, 
                model_name,
                custom_seed=seed
            )
            
            if image_path:
                target_scene['image_path'] = image_path
                
                # Return all current images
                all_images = []
                for scene in scenes:
                    if scene.get('image_path') and os.path.exists(scene['image_path']):
                        all_images.append(scene['image_path'])
                
                progress(1.0, desc="Regeneration complete!")
                return f"✅ Regenerated scene {scene_number} successfully!", all_images
            else:
                return f"❌ Failed to regenerate scene {scene_number}", []
                
        except Exception as e:
            return f"❌ Error regenerating scene: {str(e)}", []
    
    def assemble_video(scenes, output_name, progress=gr.Progress()):
        """Assemble final video from generated content"""
        try:
            if not scenes:
                return None, "❌ No scenes loaded. Please parse files first."
            
            if not output_name:
                output_name = "episode_video"
            
            # Check if we have images and/or audio
            has_images = any(scene.get('image_path') for scene in scenes)
            has_audio = any(scene.get('audio_path') for scene in scenes)
            
            if not has_images and not has_audio:
                return None, "❌ No generated content found. Please generate images and/or audio first."
            
            progress(0, desc="Assembling video...")
            video_path, message = generator.create_video(scenes, output_name)
            progress(1.0, desc="Video assembly complete!")
            
            return video_path, message
                
        except Exception as e:
            return None, f"❌ Error assembling video: {str(e)}"
    
    def test_connection():
        """Test ComfyUI connection"""
        try:
            response = requests.get(f"{generator.comfyui_url}/system_stats", timeout=5)
            if response.status_code == 200:
                return "✅ ComfyUI connection successful"
            else:
                return f"❌ ComfyUI returned status {response.status_code}"
        except Exception as e:
            return f"❌ Cannot connect to ComfyUI: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Episode Video Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 Episode Video Generator")
        gr.Markdown("Multi-step workflow: Parse files → Generate images → Generate audio → Assemble video")
        
        # Store state
        scenes_state = gr.State([])
        characters_state = gr.State({})
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Step 1: Input Files")
                script_file = gr.File(
                    label="Episode Script (.txt)",
                    file_types=[".txt"],
                    type="filepath"
                )
                prompts_file = gr.File(
                    label="Image Prompts (.txt)", 
                    file_types=[".txt"],
                    type="filepath"
                )
                
                parse_btn = gr.Button("📖 Parse Files", variant="secondary")
                parse_status = gr.Textbox(label="File Parsing Status", interactive=False, lines=8)
                
                gr.Markdown("### ⚙️ Settings")
                model_dropdown = gr.Dropdown(
                    choices=generator.get_available_models(),
                    value="disneyPixarCartoon_v10.safetensors",
                    label="AI Model for Images"
                )
                
                voice_file = gr.File(
                    label="Voice Sample (optional)",
                    file_types=[".wav", ".mp3", ".flac"],
                    type="filepath"
                )
                
                output_name = gr.Textbox(
                    value="episode_01_video",
                    label="Output Video Name"
                )
                
            with gr.Column():
                gr.Markdown("### 🎨 Step 2: Generate Images")
                generate_images_btn = gr.Button("🎨 Generate All Images", variant="primary")
                images_status = gr.Textbox(label="Image Generation Status", interactive=False)
                
                # Individual regeneration controls
                with gr.Row():
                    scene_number_input = gr.Number(
                        label="Scene Number to Regenerate", 
                        value=1, 
                        minimum=1, 
                        step=1,
                        scale=2
                    )
                    custom_seed_input = gr.Number(
                        label="Custom Seed (optional)", 
                        value=None, 
                        minimum=0, 
                        step=1,
                        scale=2
                    )
                    regenerate_btn = gr.Button("🔄 Regenerate Single Image", variant="secondary", scale=1)
                
                images_gallery = gr.Gallery(
                    label="Generated Images (Click to view full size)",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="400px"
                )
                
                gr.Markdown("### �️ Step 3: Generate Audio")
                generate_audio_btn = gr.Button("🎙️ Generate All Audio", variant="primary")
                audio_status = gr.Textbox(label="Audio Generation Status", interactive=False)
                audio_files = gr.File(label="Generated Audio Files", file_count="multiple", interactive=False)
                
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎬 Step 4: Assemble Video")
                assemble_btn = gr.Button("🎬 Assemble Final Video", variant="primary", size="lg")
                
            with gr.Column():
                gr.Markdown("### 🔧 Connection Test")
                test_btn = gr.Button("Test ComfyUI Connection")
                connection_status = gr.Textbox(label="Connection Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Final Output")
                video_output = gr.Video(label="Generated Video")
                final_status = gr.Textbox(label="Final Assembly Status", interactive=False)
        
        # Event handlers
        parse_btn.click(
            fn=parse_files,
            inputs=[script_file, prompts_file],
            outputs=[parse_status, scenes_state, characters_state]
        )
        
        generate_images_btn.click(
            fn=generate_images_only,
            inputs=[scenes_state, characters_state, model_dropdown],
            outputs=[images_status, images_gallery]
        )
        
        regenerate_btn.click(
            fn=regenerate_single_image,
            inputs=[scenes_state, characters_state, scene_number_input, model_dropdown, custom_seed_input],
            outputs=[images_status, images_gallery]
        )
        
        generate_audio_btn.click(
            fn=generate_audio_only,
            inputs=[scenes_state, voice_file],
            outputs=[audio_status, audio_files]
        )
        
        assemble_btn.click(
            fn=assemble_video,
            inputs=[scenes_state, output_name],
            outputs=[video_output, final_status]
        )
        
        test_btn.click(
            fn=test_connection,
            outputs=[connection_status]
        )
    
    return app


if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )