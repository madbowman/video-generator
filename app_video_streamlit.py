#!/usr/bin/env python3
"""
Video Generator - Streamlit Version
Uses same technology as the image folder with timeline view
"""

import streamlit as st
import os
import re
import json
import time
import requests
import io
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from scipy.io import wavfile
import subprocess
import tempfile

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
        else:
            self.device = "cpu"
            
        # Create temp directories
        self.temp_audio_dir = self.output_dir / "temp_audio"
        self.temp_image_dir = self.output_dir / "temp_images"
        self.temp_audio_dir.mkdir(exist_ok=True)
        self.temp_image_dir.mkdir(exist_ok=True)
    
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
    
    def parse_episode_files(self, script_file, prompts_file):
        """Parse both script and prompts files to extract scenes"""
        scenes = []
        characters = {}
        
        try:
            # Parse script file for narrative text
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
                
            # Parse prompts file for visual descriptions and characters
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts_content = f.read()
            
            # Extract characters from prompts file
            characters = self._extract_characters(prompts_content)
            
            # Extract scenes by matching script and prompts
            scenes = self._extract_scenes(script_content, prompts_content)
            
            return scenes, characters
            
        except Exception as e:
            st.error(f"Error parsing files: {e}")
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
                'duration': 5.0,  # Default 5 seconds per scene
                'image_path': None,
                'audio_path': None
            }
            scenes.append(scene_data)
        
        return scenes
    
    def create_workflow(self, prompt, model_name, seed, width=1024, height=1024):
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
    
    def generate_single_image(self, scene, model_name, characters, custom_seed=None):
        """Generate image for a single scene"""
        try:
            scene_number = scene['number']
            
            # Combine scene prompts with character descriptions
            if not scene['image_prompts']:
                base_prompt = f"Pixar animation style scene: {scene['title']}"
            else:
                base_prompt = scene['image_prompts'][0]
            
            # Add character descriptions to improve consistency
            character_context = ""
            for char_name, char_data in characters.items():
                if char_name.lower() in base_prompt.lower():
                    character_context += f" {char_data.get('full_description', '')}"
            
            full_prompt = f"{base_prompt} {character_context}".strip()
            
            # Use custom seed or generate one based on scene and time
            seed = custom_seed if custom_seed else int(time.time() * 1000) + scene_number
            
            # Create ComfyUI workflow
            workflow = self.create_workflow(full_prompt, model_name, seed)
            
            # Submit to ComfyUI and wait for result
            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            
            prompt_id = response.json()["prompt_id"]
            
            # Wait for completion with progress tracking
            max_wait = 180  # 3 minutes
            wait_time = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while wait_time < max_wait:
                progress = wait_time / max_wait
                progress_bar.progress(progress)
                status_text.text(f"Generating scene {scene_number}... {wait_time}s/{max_wait}s")
                
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
                                        
                                        progress_bar.progress(1.0)
                                        status_text.text(f"✅ Scene {scene_number} completed!")
                                        
                                        return str(output_path)
                            break
                except Exception as e:
                    continue
            
            progress_bar.progress(1.0)
            status_text.text(f"❌ Timeout waiting for scene {scene_number}")
            return None
            
        except Exception as e:
            st.error(f"Error generating image for scene {scene_number}: {e}")
            return None
    
    def generate_scene_audio(self, scene_text, scene_number, voice_file=None):
        """Generate TTS audio for a scene"""
        if not AUDIO_AVAILABLE:
            st.warning("Audio generation not available")
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
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"Generating audio for scene {scene_number}...")
            
            self.narrator.narrate_story(
                story_text=clean_text,
                output_path=str(output_path)
            )
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Audio for scene {scene_number} completed!")
            
            return str(output_path)
            
        except Exception as e:
            st.error(f"Error generating audio for scene {scene_number}: {e}")
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
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, scene in enumerate(scenes):
                progress = i / len(scenes)
                progress_bar.progress(progress)
                status_text.text(f"Creating video segment {scene['number']}...")
                
                scene_num = scene['number']
                duration = scene.get('duration', 5.0)
                
                image_path = scene.get('image_path')
                audio_path = scene.get('audio_path')
                segment_path = os.path.join(temp_dir, f"segment_{scene_num:03d}.mp4")
                
                # Check if files exist
                if not image_path or not os.path.exists(image_path):
                    st.warning(f"Missing image for scene {scene_num}")
                    continue
                
                # Create video segment for this scene
                cmd = ['ffmpeg', '-y', '-loop', '1', '-i', image_path]
                
                if audio_path and os.path.exists(audio_path):
                    # Use audio duration if available
                    cmd.extend(['-i', audio_path])
                    cmd.extend(['-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-pix_fmt', 'yuv420p'])
                else:
                    # Use specified duration
                    cmd.extend(['-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p'])
                
                cmd.extend(['-r', str(fps), segment_path])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    video_segments.append(segment_path)
                else:
                    st.error(f"Error creating segment {scene_num}: {result.stderr}")
            
            if not video_segments:
                progress_bar.progress(1.0)
                status_text.text("❌ No video segments were created successfully")
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
            
            status_text.text("Combining all segments...")
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            
            progress_bar.progress(1.0)
            
            if result.returncode == 0:
                # Cleanup temp files
                for segment in video_segments:
                    try:
                        os.remove(segment)
                    except:
                        pass
                try:
                    os.remove(concat_file)
                    os.rmdir(temp_dir)
                except:
                    pass
                
                status_text.text(f"✅ Video created successfully!")
                return str(output_path), f"✅ Video created: {output_path}"
            else:
                status_text.text(f"❌ Error combining segments")
                return None, f"❌ Error combining segments: {result.stderr}"
                
        except Exception as e:
            return None, f"❌ Error creating video: {str(e)}"

def main():
    st.set_page_config(
        page_title="🎬 Video Generator",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'generator' not in st.session_state:
        st.session_state.generator = VideoGenerator()
    
    if 'scenes' not in st.session_state:
        st.session_state.scenes = []
    
    if 'characters' not in st.session_state:
        st.session_state.characters = {}
    
    if 'audio_generation_state' not in st.session_state:
        st.session_state.audio_generation_state = {
            'is_running': False,
            'is_paused': False,
            'current_scene': 0,
            'total_scenes': 0,
            'progress': 0.0,
            'start_time': None,
            'pause_time': None
        }
    
    if 'image_generation_state' not in st.session_state:
        st.session_state.image_generation_state = {
            'is_running': False,
            'is_paused': False,
            'current_scene': 0,
            'total_scenes': 0,
            'progress': 0.0,
            'start_time': None,
            'pause_time': None
        }
    
    # Auto session management
    if 'session_auto_loaded' not in st.session_state:
        st.session_state.session_auto_loaded = False
    
    def auto_save_session():
        """Auto-save current session state"""
        if st.session_state.scenes:  # Only save if we have data
            session_data = {
                'scenes': st.session_state.scenes,
                'characters': st.session_state.characters,
                'voice_file_path': getattr(st.session_state, 'voice_file_path', None),
                'timestamp': time.time()
            }
            
            auto_session_file = st.session_state.generator.output_dir / "auto_session.json"
            try:
                with open(auto_session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                pass  # Silently fail to avoid disrupting UI
    
    def auto_load_session():
        """Auto-load the most recent session on startup"""
        if st.session_state.session_auto_loaded:
            return None
        
        auto_session_file = st.session_state.generator.output_dir / "auto_session.json"
        
        if auto_session_file.exists():
            try:
                with open(auto_session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                scenes = session_data.get('scenes', [])
                if scenes:  # Only load if there's actual data
                    st.session_state.scenes = scenes
                    st.session_state.characters = session_data.get('characters', {})
                    st.session_state.voice_file_path = session_data.get('voice_file_path')
                    
                    # Validate existing file paths in scenes and count valid ones
                    valid_images = 0
                    valid_audio = 0
                    
                    for scene in scenes:
                        # Validate image path
                        if scene.get('image_path'):
                            if os.path.exists(scene['image_path']):
                                valid_images += 1
                            else:
                                # Try to fix relative path issue
                                if not os.path.isabs(scene['image_path']):
                                    abs_path = st.session_state.generator.output_dir / scene['image_path']
                                    if abs_path.exists():
                                        scene['image_path'] = str(abs_path)
                                        valid_images += 1
                        
                        # Validate audio path
                        if scene.get('audio_path'):
                            if os.path.exists(scene['audio_path']):
                                valid_audio += 1
                            else:
                                # Try to fix relative path issue
                                if not os.path.isabs(scene['audio_path']):
                                    abs_path = st.session_state.generator.output_dir / scene['audio_path']
                                    if abs_path.exists():
                                        scene['audio_path'] = str(abs_path)
                                        valid_audio += 1
                    
                    # Auto-discover any additional files not already linked
                    discovered_files = auto_discover_files()
                    
                    st.session_state.session_auto_loaded = True
                    
                    total_images = valid_images + discovered_files['images']
                    total_audio = valid_audio + discovered_files['audio']
                    
                    message_parts = [f"{len(scenes)} scenes"]
                    if total_images > 0:
                        message_parts.append(f"{total_images} images")
                    if total_audio > 0:
                        message_parts.append(f"{total_audio} audio files")
                    
                    return f"🔄 Auto-restored session with {', '.join(message_parts)}"
            except Exception as e:
                pass  # Silently fail
        
        st.session_state.session_auto_loaded = True
        return None
    
    def auto_discover_files():
        """Scan output directory for existing files and link them to scenes"""
        output_dir = st.session_state.generator.output_dir
        loaded_images = 0
        loaded_audio = 0
        
        # Search directories to check
        search_dirs = [
            output_dir,  # Main output directory
            output_dir / "temp_images",  # Image subdirectory
            output_dir / "temp_audio",   # Audio subdirectory
        ]
        
        for scene in st.session_state.scenes:
            scene_number = scene['number']
            
            # Look for image files
            if not (scene.get('image_path') and os.path.exists(scene.get('image_path', ''))):
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                    for ext in ['.png', '.jpg', '.jpeg']:
                        for pattern in [
                            f"scene_{scene_number:03d}*{ext}",  # 3-digit format (001, 002)
                            f"scene_{scene_number:02d}*{ext}",  # 2-digit format (01, 02)
                            f"scene_{scene_number}*{ext}",      # No padding (1, 2)
                            f"img_{scene_number:03d}*{ext}",
                            f"img_{scene_number:02d}*{ext}",
                            f"img_{scene_number}*{ext}",
                            f"image_{scene_number:03d}*{ext}",
                            f"image_{scene_number:02d}*{ext}",
                            f"image_{scene_number}*{ext}"
                        ]:
                            matching_files = list(search_dir.glob(pattern))
                            if matching_files:
                                image_path = max(matching_files, key=lambda p: p.stat().st_mtime)
                                if image_path.exists():
                                    scene['image_path'] = str(image_path)
                                    loaded_images += 1
                                    break
                        if scene.get('image_path'):
                            break
                    if scene.get('image_path'):
                        break
            
            # Look for audio files
            if not (scene.get('audio_path') and os.path.exists(scene.get('audio_path', ''))):
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                    for ext in ['.wav', '.mp3', '.m4a']:
                        for pattern in [
                            f"scene_{scene_number:03d}*{ext}",  # 3-digit format (001, 002)
                            f"scene_{scene_number:02d}*{ext}",  # 2-digit format (01, 02)
                            f"scene_{scene_number}*{ext}",      # No padding (1, 2)
                            f"audio_{scene_number:03d}*{ext}",
                            f"audio_{scene_number:02d}*{ext}",
                            f"audio_{scene_number}*{ext}"
                        ]:
                            matching_files = list(search_dir.glob(pattern))
                            if matching_files:
                                audio_path = max(matching_files, key=lambda p: p.stat().st_mtime)
                                if audio_path.exists():
                                    scene['audio_path'] = str(audio_path)
                                    loaded_audio += 1
                                    break
                        if scene.get('audio_path'):
                            break
                    if scene.get('audio_path'):
                        break
        
        return {'images': loaded_images, 'audio': loaded_audio}
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # ComfyUI connection test
        if st.button("🔌 Test ComfyUI Connection", key="test_comfyui"):
            try:
                response = requests.get(f"{st.session_state.generator.comfyui_url}/system_stats", timeout=5)
                if response.status_code == 200:
                    st.success("✅ ComfyUI connected successfully")
                else:
                    st.error(f"❌ ComfyUI returned status {response.status_code}")
            except Exception as e:
                st.error(f"❌ Cannot connect to ComfyUI: {str(e)}")
        
        # Model selection
        available_models = st.session_state.generator.get_available_models()
        selected_model = st.selectbox(
            "🎨 AI Model",
            available_models,
            index=0 if available_models else None
        )
        
        # Voice file upload with persistent storage
        st.markdown("### 🎙️ Voice Settings")
        voice_file = st.file_uploader(
            "Voice Sample (Optional)",
            type=['wav', 'mp3', 'flac'],
            help="Upload a voice sample for voice cloning. Will be saved for the session."
        )
        
        # Save voice file persistently
        if voice_file is not None:
            voice_dir = st.session_state.generator.output_dir / "voice_samples"
            voice_dir.mkdir(exist_ok=True)
            
            voice_path = voice_dir / f"session_voice_{voice_file.name}"
            
            # Only save if file doesn't exist or is different
            if not voice_path.exists():
                with open(voice_path, 'wb') as f:
                    f.write(voice_file.getvalue())
                st.success(f"✅ Voice sample saved: {voice_file.name}")
            
            # Store in session state
            st.session_state.voice_file_path = str(voice_path)
            st.info(f"🎙️ Using voice: {voice_file.name}")
        else:
            st.session_state.voice_file_path = None
        
        # Show current voice status
        if hasattr(st.session_state, 'voice_file_path') and st.session_state.voice_file_path:
            if os.path.exists(st.session_state.voice_file_path):
                st.success("🎙️ Voice sample ready")
                if st.button("🗑️ Clear Voice Sample", key="clear_voice"):
                    try:
                        os.remove(st.session_state.voice_file_path)
                    except:
                        pass
                    st.session_state.voice_file_path = None
                    st.rerun()
            else:
                st.warning("⚠️ Voice file missing")
                st.session_state.voice_file_path = None
        
        # Output settings
        output_name = st.text_input("📹 Output Video Name", value="episode_video")
        
        st.markdown("### 💾 Session Management")
        
        # Save/Load session
        if st.button("💾 Save Session", key="save_session"):
            session_data = {
                'scenes': st.session_state.scenes,
                'characters': st.session_state.characters,
                'voice_file_path': getattr(st.session_state, 'voice_file_path', None),
                'model': selected_model,
                'output_name': output_name
            }
            
            session_file = st.session_state.generator.output_dir / f"session_{int(time.time())}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Session saved: {session_file.name}")
        
        # Load session
        session_files = list(st.session_state.generator.output_dir.glob("session_*.json"))
        if session_files:
            selected_session = st.selectbox(
                "Load Session",
                [""] + [f.name for f in sorted(session_files, reverse=True)],
                help="Load a previously saved session"
            )
            
            if selected_session and st.button("📂 Load Session", key="load_session"):
                session_file = st.session_state.generator.output_dir / selected_session
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    st.session_state.scenes = session_data.get('scenes', [])
                    st.session_state.characters = session_data.get('characters', {})
                    st.session_state.voice_file_path = session_data.get('voice_file_path')
                    
                    st.success(f"✅ Session loaded: {selected_session}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading session: {e}")
        
        # Cleanup old temp files
        if st.button("🧹 Cleanup Temp Files", key="cleanup_temp"):
            temp_files_removed = 0
            audio_files_removed = 0
            voice_files_removed = 0
            
            # Remove old temp voice files (root directory)
            for temp_file in Path(".").glob("temp_voice_*.wav"):
                try:
                    os.remove(temp_file)
                    temp_files_removed += 1
                except:
                    pass
            
            # Remove temp voice files from output directory
            for temp_file in st.session_state.generator.output_dir.glob("temp_voice_*.wav"):
                try:
                    os.remove(temp_file)
                    temp_files_removed += 1
                except:
                    pass
            
            # Remove old temp image files without corresponding scenes
            scene_numbers = [s['number'] for s in st.session_state.scenes] if st.session_state.scenes else []
            for temp_file in st.session_state.generator.temp_image_dir.glob("scene_*.png"):
                try:
                    scene_num = int(temp_file.stem.split('_')[1])
                    if scene_num not in scene_numbers:
                        os.remove(temp_file)
                        temp_files_removed += 1
                except:
                    pass
            
            # Remove ALL audio files from temp_audio directory
            temp_audio_dir = st.session_state.generator.temp_audio_dir
            if temp_audio_dir.exists():
                for audio_file in temp_audio_dir.glob("scene_*.wav"):
                    try:
                        os.remove(audio_file)
                        audio_files_removed += 1
                    except:
                        pass
            
            # Remove voice samples (with confirmation)
            voice_samples_dir = st.session_state.generator.output_dir / "voice_samples"
            if voice_samples_dir.exists():
                for voice_file in voice_samples_dir.glob("session_voice_*"):
                    try:
                        os.remove(voice_file)
                        voice_files_removed += 1
                    except:
                        pass
            
            # Clear current voice file path from session
            if hasattr(st.session_state, 'voice_file_path'):
                st.session_state.voice_file_path = None
            
            # Reset audio generation state
            st.session_state.audio_generation_state = {
                'is_running': False,
                'is_paused': False,
                'current_scene': 0,
                'total_scenes': 0,
                'progress': 0.0,
                'start_time': None,
                'pause_time': None
            }
            
            # Clear audio paths from scenes
            for scene in st.session_state.scenes:
                scene['audio_path'] = None
            
            total_removed = temp_files_removed + audio_files_removed + voice_files_removed
            
            if total_removed > 0:
                st.success(f"✅ Cleanup Complete!")
                st.info(f"🗑️ Removed {temp_files_removed} temp files, {audio_files_removed} audio files, {voice_files_removed} voice samples")
            else:
                st.info("ℹ️ No files to cleanup")
    
    # Main interface
    st.title("🎬 Episode Video Generator")
    st.markdown("Create MP4 videos from episode scripts with AI-generated images and TTS audio")
    
    # Auto-load previous session on startup
    if not st.session_state.session_auto_loaded:
        auto_load_message = auto_load_session()
        if auto_load_message:
            st.success(auto_load_message)
    
    # Navigation using radio buttons (persists across reruns)
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "📄 Import"
    
    active_tab = st.radio(
        "Navigation",
        ["📄 Import", "🎨 Timeline & Production", "🎬 Video"],
        index=["📄 Import", "🎨 Timeline & Production", "🎬 Video"].index(st.session_state.active_tab),
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.active_tab = active_tab
    
    st.divider()
    
    if active_tab == "📄 Import":
        st.header("📄 Import Episode Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            script_file = st.file_uploader(
                "Episode Script (.txt)",
                type=['txt'],
                help="Upload your episode script with [SCENE X: TITLE] markers"
            )
        
        with col2:
            prompts_file = st.file_uploader(
                "Image Prompts (.txt)",
                type=['txt'],
                help="Upload your image prompts file with CHARACTER REFERENCE and IMAGE sections"
            )
        
        if st.button("📖 Parse Files", type="primary", key="parse_files"):
            if script_file and prompts_file:
                # Save uploaded files temporarily
                script_path = f"temp_script_{int(time.time())}.txt"
                prompts_path = f"temp_prompts_{int(time.time())}.txt"
                
                with open(script_path, 'wb') as f:
                    f.write(script_file.getvalue())
                
                with open(prompts_path, 'wb') as f:
                    f.write(prompts_file.getvalue())
                
                # Parse files
                scenes, characters = st.session_state.generator.parse_episode_files(script_path, prompts_path)
                
                # Clean up temp files
                os.remove(script_path)
                os.remove(prompts_path)
                
                # Store in session state
                st.session_state.scenes = scenes
                st.session_state.characters = characters
                
                # Auto-discover existing files for newly parsed scenes
                loaded_files = auto_discover_files()
                
                # Auto-save the session
                auto_save_session()
                
                if scenes:
                    success_msg = f"✅ Successfully parsed {len(scenes)} scenes and {len(characters)} characters"
                    if loaded_files['images'] > 0 or loaded_files['audio'] > 0:
                        success_msg += f" (Found {loaded_files['images']} existing images, {loaded_files['audio']} audio files)"
                    st.success(success_msg)
                    
                    # Show preview
                    with st.expander("📋 Parsed Content Preview"):
                        st.write("**Characters:**")
                        for char_name in characters.keys():
                            st.write(f"• {char_name}")
                        
                        st.write("**Scenes:**")
                        for scene in scenes[:5]:  # Show first 5
                            st.write(f"• Scene {scene['number']}: {scene['title']}")
                        if len(scenes) > 5:
                            st.write(f"... and {len(scenes) - 5} more scenes")
                else:
                    st.error("❌ No scenes found in the files")
            else:
                st.error("❌ Please upload both script and prompts files")
    
    elif active_tab == "🎨 Timeline & Production":
        st.header("🎨 Timeline & Production")
        st.caption("Generate and manage images and audio for each scene")
        
        if not st.session_state.scenes:
            st.info("📄 Please import and parse episode files first")
        else:
            # Top controls with pause/resume functionality
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                image_state = st.session_state.image_generation_state
                
                if image_state['is_running']:
                    st.write(f"**🎨 Generating Images:** Scene {image_state['current_scene']}/{image_state['total_scenes']}")
                    
                    # Progress bar
                    progress_value = image_state['current_scene'] / image_state['total_scenes'] if image_state['total_scenes'] > 0 else 0
                    st.progress(progress_value)
                    
                    # Estimated time
                    if image_state['start_time']:
                        elapsed = time.time() - image_state['start_time']
                        if image_state['current_scene'] > 0:
                            avg_time_per_scene = elapsed / image_state['current_scene']
                            remaining_scenes = image_state['total_scenes'] - image_state['current_scene']
                            est_remaining = avg_time_per_scene * remaining_scenes
                            st.caption(f"⏱️ Estimated remaining: {int(est_remaining/60)}m {int(est_remaining%60)}s")
                else:
                    # Count pending scenes
                    pending_scenes = sum(1 for scene in st.session_state.scenes 
                                       if not (scene.get('image_path') and os.path.exists(scene.get('image_path', ''))))
                    completed_scenes = len(st.session_state.scenes) - pending_scenes
                    
                    st.write(f"**Image Overview:** {completed_scenes}/{len(st.session_state.scenes)} scenes completed")
                    if pending_scenes > 0:
                        st.info(f"📋 {pending_scenes} scenes need image generation")
                    else:
                        st.success("✅ All scenes have images!")
            
            with col2:
                if not image_state['is_running']:
                    if st.button("🎨 Start/Resume Images", type="primary", key="img_start_resume"):
                        image_state['is_running'] = True
                        image_state['is_paused'] = False
                        image_state['total_scenes'] = len(st.session_state.scenes)
                        image_state['start_time'] = time.time()
                        
                        # Find first scene without image
                        for i, scene in enumerate(st.session_state.scenes):
                            if not (scene.get('image_path') and os.path.exists(scene.get('image_path', ''))):
                                image_state['current_scene'] = i
                                break
                        else:
                            image_state['current_scene'] = len(st.session_state.scenes)
                        
                        st.rerun()
                else:
                    if st.button("⏸️ Pause", type="secondary", key="img_pause"):
                        image_state['is_running'] = False
                        image_state['is_paused'] = True
                        image_state['pause_time'] = time.time()
                        st.success("⏸️ Image generation paused. Click Resume to continue.")
                        st.rerun()
            
            with col3:
                if image_state['is_paused']:
                    if st.button("▶️ Resume", type="primary", key="img_resume"):
                        image_state['is_running'] = True
                        image_state['is_paused'] = False
                        # Adjust start time to account for pause
                        if image_state['pause_time'] and image_state['start_time']:
                            pause_duration = time.time() - image_state['pause_time']
                            image_state['start_time'] += pause_duration
                        st.rerun()
                
                if st.button("🛑 Stop & Reset", key="img_stop_reset"):
                    image_state['is_running'] = False
                    image_state['is_paused'] = False
                    image_state['current_scene'] = 0
                    image_state['progress'] = 0.0
                    image_state['start_time'] = None
                    st.success("🛑 Image generation stopped and reset.")
                    st.rerun()
            
            with col4:
                if st.button("🔄 Refresh View", key="img_refresh"):
                    st.rerun()
            
            # Auto-generation logic when running
            if image_state['is_running'] and not image_state['is_paused']:
                # Find next scene that needs image
                scenes_to_process = []
                for i, scene in enumerate(st.session_state.scenes):
                    if not (scene.get('image_path') and os.path.exists(scene.get('image_path', ''))):
                        scenes_to_process.append((i, scene))
                
                if scenes_to_process:
                    # Process next scene
                    scene_idx, scene = scenes_to_process[0]
                    image_state['current_scene'] = scene_idx + 1
                    
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    with progress_placeholder.container():
                        st.info(f"🎨 Generating image for Scene {scene['number']}: {scene['title']}")
                    
                    with status_placeholder.container():
                        with st.spinner(f"Processing scene {scene['number']}..."):
                            image_path = st.session_state.generator.generate_single_image(
                                scene, selected_model, st.session_state.characters
                            )
                            
                            if image_path:
                                scene['image_path'] = image_path
                                # Auto-save session when new image is generated
                                auto_save_session()
                    
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    
                    # Check if we're done
                    remaining = len([s for s in st.session_state.scenes 
                                   if not (s.get('image_path') and os.path.exists(s.get('image_path', '')))])
                    
                    if remaining == 0:
                        image_state['is_running'] = False
                        image_state['is_paused'] = False
                        st.success("🎉 All image generation completed!")
                        st.balloons()
                    
                    # Auto-refresh to continue with next scene
                    time.sleep(1)
                    st.rerun()
                else:
                    # All scenes processed
                    image_state['is_running'] = False
                    image_state['is_paused'] = False
                    st.success("✅ All scenes have images!")
            
            st.divider()
            
            # Image generation statistics
            if st.session_state.scenes:
                image_stats_col1, image_stats_col2, image_stats_col3 = st.columns(3)
                
                with image_stats_col1:
                    completed = sum(1 for s in st.session_state.scenes 
                                  if s.get('image_path') and os.path.exists(s.get('image_path', '')))
                    st.metric("Completed Images", f"{completed}/{len(st.session_state.scenes)}")
                
                with image_stats_col2:
                    if image_state['start_time'] and completed > 0:
                        elapsed = time.time() - image_state['start_time']
                        avg_per_scene = elapsed / completed
                        st.metric("Avg Time/Scene", f"{avg_per_scene:.1f}s")
                    else:
                        st.metric("Avg Time/Scene", "--")
                
                with image_stats_col3:
                    # Calculate total file size of generated images
                    total_size = 0
                    for scene in st.session_state.scenes:
                        if scene.get('image_path') and os.path.exists(scene.get('image_path', '')):
                            try:
                                size_bytes = os.path.getsize(scene['image_path'])
                                total_size += size_bytes
                            except:
                                pass
                    
                    total_mb = total_size / (1024 * 1024)
                    st.metric("Total Images Size", f"{total_mb:.1f} MB")
            
            st.divider()
            
            # Scene timeline with editable fields (similar to image app)
            for i, scene in enumerate(st.session_state.scenes):
                scene_key = f"scene_{scene['number']}"
                
                with st.container():
                    # Scene header with status indicators
                    col_header1, col_header2, col_header3 = st.columns([4, 1, 1])
                    
                    with col_header1:
                        st.subheader(f"Scene {scene['number']}: {scene['title']}")
                    
                    with col_header2:
                        has_image = scene.get('image_path') and os.path.exists(scene.get('image_path', ''))
                        if has_image:
                            st.success("🖼️ Image ✅")
                        else:
                            st.warning("🖼️ No image")
                    
                    with col_header3:
                        has_audio = scene.get('audio_path') and os.path.exists(scene.get('audio_path', ''))
                        if has_audio:
                            # Calculate and display actual audio duration
                            try:
                                import wave
                                with wave.open(scene['audio_path'], 'r') as wav_file:
                                    frames = wav_file.getnframes()
                                    rate = wav_file.getframerate()
                                    duration = frames / float(rate)
                                    scene['duration'] = duration  # Store actual audio duration
                                    st.success(f"🎙️ Audio {duration:.1f}s")
                            except:
                                st.success("🎙️ Audio ✅")
                        else:
                            st.warning("🎙️ No audio")
                    
                    # Main content area - 3 columns for script, image, and audio
                    col_content1, col_content2, col_content3 = st.columns([2, 2, 2])
                    
                    with col_content1:
                        st.write("**Script Text:**")
                        script_text = st.text_area(
                            "Script",
                            value=scene['script_text'],
                            height=200,
                            key=f"script_edit_{scene['number']}",
                            label_visibility="collapsed"
                        )
                        scene['script_text'] = script_text
                        
                        st.write("**Image Prompt:**")
                        current_prompt = scene['image_prompts'][0] if scene['image_prompts'] else ""
                        image_prompt = st.text_area(
                            "Image Prompt",
                            value=current_prompt,
                            height=200,
                            key=f"prompt_edit_{scene['number']}",
                            label_visibility="collapsed"
                        )
                        scene['image_prompts'] = [image_prompt] if image_prompt else scene['image_prompts']
                    
                    with col_content2:
                        st.write("**Image:**")
                        # Image display and generation controls
                        if has_image:
                            st.image(scene['image_path'], caption=f"Scene {scene['number']}", use_container_width=True)
                        else:
                            st.info("No image generated yet")
                        
                        # Individual image generation controls
                        if st.button(f"🎨 Generate Image", key=f"gen_{scene['number']}", use_container_width=True):
                            # Delete existing image file if it exists
                            if scene.get('image_path') and os.path.exists(scene['image_path']):
                                try:
                                    os.remove(scene['image_path'])
                                except:
                                    pass
                                scene['image_path'] = None
                            
                            with st.spinner(f"Generating scene {scene['number']}..."):
                                image_path = st.session_state.generator.generate_single_image(
                                    scene, selected_model, st.session_state.characters
                                )
                                
                                if image_path:
                                    scene['image_path'] = image_path
                                    auto_save_session()
                                    st.success(f"✅ Generated!")
                                    st.rerun()
                    
                    with col_content3:
                        st.write("**Audio:**")
                        # Audio playback
                        if has_audio:
                            st.audio(scene['audio_path'])
                            try:
                                import wave
                                with wave.open(scene['audio_path'], 'r') as wav_file:
                                    frames = wav_file.getnframes()
                                    rate = wav_file.getframerate()
                                    duration = frames / float(rate)
                                    st.caption(f"⏱️ {duration:.1f}s")
                            except:
                                pass
                        else:
                            st.info("No audio generated yet")
                        
                        # Audio generation controls
                        if AUDIO_AVAILABLE:
                            if st.button(f"🎙️ Generate Audio", key=f"gen_audio_{scene['number']}", use_container_width=True):
                                # Delete existing audio file if it exists
                                if scene.get('audio_path') and os.path.exists(scene['audio_path']):
                                    try:
                                        os.remove(scene['audio_path'])
                                    except:
                                        pass
                                    scene['audio_path'] = None
                                
                                voice_path = getattr(st.session_state, 'voice_file_path', None)
                                
                                with st.spinner(f"Generating audio for scene {scene['number']}..."):
                                    audio_path = st.session_state.generator.generate_scene_audio(
                                        scene.get('audio_script', scene['script_text']), 
                                        scene['number'], 
                                        voice_path
                                    )
                                    
                                    if audio_path:
                                        scene['audio_path'] = audio_path
                                        auto_save_session()
                                        st.success(f"✅ Generated!")
                                        st.rerun()
                        else:
                            st.warning("Audio system not available")
    
    elif active_tab == "🎬 Video":
        st.header("🎬 Video Assembly")
        
        if not st.session_state.scenes:
            st.info("📄 Please import and parse episode files first")
        else:
            # Check status
            has_images = sum(1 for s in st.session_state.scenes if s.get('image_path') and os.path.exists(s.get('image_path', '')))
            has_audio = sum(1 for s in st.session_state.scenes if s.get('audio_path') and os.path.exists(s.get('audio_path', '')))
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Assembly Status")
                st.metric("Scenes with Images", f"{has_images}/{len(st.session_state.scenes)}")
                st.metric("Scenes with Audio", f"{has_audio}/{len(st.session_state.scenes)}")
                
                if has_images > 0:
                    if st.button("🎬 Create Video", type="primary"):
                        video_path, message = st.session_state.generator.create_video(
                            st.session_state.scenes, output_name
                        )
                        
                        if video_path:
                            st.success(message)
                            st.session_state.final_video = video_path
                        else:
                            st.error(message)
                else:
                    st.warning("⚠️ Generate at least some images before creating video")
            
            with col2:
                st.subheader("Final Video")
                
                if hasattr(st.session_state, 'final_video') and os.path.exists(st.session_state.final_video):
                    st.video(st.session_state.final_video)
                    
                    # Download button
                    with open(st.session_state.final_video, 'rb') as f:
                        st.download_button(
                            label="📥 Download Video",
                            data=f.read(),
                            file_name=f"{output_name}.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.info("Video will appear here after assembly")

if __name__ == "__main__":
    main()
