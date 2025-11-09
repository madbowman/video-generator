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
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # ComfyUI connection test
        if st.button("🔌 Test ComfyUI Connection"):
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
                if st.button("🗑️ Clear Voice Sample"):
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
        if st.button("💾 Save Session"):
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
            
            if selected_session and st.button("📂 Load Session"):
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
        if st.button("🧹 Cleanup Temp Files"):
            temp_files_removed = 0
            
            # Remove old temp voice files
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
            
            st.success(f"✅ Removed {temp_files_removed} temporary files")
    
    # Main interface
    st.title("🎬 Episode Video Generator")
    st.markdown("Create MP4 videos from episode scripts with AI-generated images and TTS audio")
    
    # Tabs for different steps
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Import", 
        "🎨 Timeline & Images", 
        "🎙️ Audio", 
        "🎬 Video"
    ])
    
    with tab1:
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
        
        if st.button("📖 Parse Files", type="primary"):
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
                
                if scenes:
                    st.success(f"✅ Successfully parsed {len(scenes)} scenes and {len(characters)} characters")
                    
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
    
    with tab2:
        st.header("🎨 Timeline & Image Generation")
        
        if not st.session_state.scenes:
            st.info("📄 Please import and parse episode files first")
        else:
            # Top controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Episode Overview:** {len(st.session_state.scenes)} scenes found")
            
            with col2:
                if st.button("🎨 Generate All Images", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, scene in enumerate(st.session_state.scenes):
                        progress = i / len(st.session_state.scenes)
                        progress_bar.progress(progress)
                        status_text.text(f"Generating scene {scene['number']}: {scene['title']}")
                        
                        image_path = st.session_state.generator.generate_single_image(
                            scene, selected_model, st.session_state.characters
                        )
                        
                        if image_path:
                            scene['image_path'] = image_path
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ All images generated!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Refresh View"):
                    st.rerun()
            
            st.divider()
            
            # Scene timeline with editable fields (similar to image app)
            for i, scene in enumerate(st.session_state.scenes):
                scene_key = f"scene_{scene['number']}"
                
                with st.container():
                    # Scene header with status indicators
                    col_header1, col_header2, col_header3, col_header4 = st.columns([3, 1, 1, 1])
                    
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
                            st.success("🎙️ Audio ✅")
                        else:
                            st.warning("🎙️ No audio")
                    
                    with col_header4:
                        duration = st.number_input(
                            "Duration (s)",
                            min_value=1.0,
                            max_value=30.0,
                            value=scene.get('duration', 5.0),
                            step=0.5,
                            key=f"duration_{scene['number']}",
                            label_visibility="collapsed"
                        )
                        scene['duration'] = duration
                    
                    # Main content area
                    col_content1, col_content2, col_content3 = st.columns([2, 2, 1])
                    
                    with col_content1:
                        st.write("**Script Text:**")
                        script_text = st.text_area(
                            "Script",
                            value=scene['script_text'],
                            height=120,
                            key=f"script_edit_{scene['number']}",
                            label_visibility="collapsed"
                        )
                        scene['script_text'] = script_text
                        
                        st.write("**Image Prompt:**")
                        current_prompt = scene['image_prompts'][0] if scene['image_prompts'] else ""
                        image_prompt = st.text_area(
                            "Image Prompt",
                            value=current_prompt,
                            height=80,
                            key=f"prompt_edit_{scene['number']}",
                            label_visibility="collapsed"
                        )
                        scene['image_prompts'] = [image_prompt] if image_prompt else scene['image_prompts']
                    
                    with col_content2:
                        # Image display and generation controls
                        if has_image:
                            st.image(scene['image_path'], caption=f"Scene {scene['number']}", use_container_width=True)
                        else:
                            st.info("No image generated yet")
                        
                        # Individual generation controls
                        col_gen1, col_gen2 = st.columns(2)
                        
                        with col_gen1:
                            if st.button(f"🎨 Generate", key=f"gen_{scene['number']}", use_container_width=True):
                                with st.spinner(f"Generating scene {scene['number']}..."):
                                    image_path = st.session_state.generator.generate_single_image(
                                        scene, selected_model, st.session_state.characters
                                    )
                                    
                                    if image_path:
                                        scene['image_path'] = image_path
                                        st.success(f"✅ Generated!")
                                        st.rerun()
                        
                        with col_gen2:
                            custom_seed = st.number_input(
                                "Seed",
                                value=None,
                                min_value=0,
                                key=f"seed_{scene['number']}",
                                label_visibility="collapsed",
                                placeholder="Random"
                            )
                            
                            if st.button(f"🔄 Regen", key=f"regen_{scene['number']}", use_container_width=True):
                                with st.spinner(f"Regenerating scene {scene['number']}..."):
                                    image_path = st.session_state.generator.generate_single_image(
                                        scene, selected_model, st.session_state.characters, custom_seed
                                    )
                                    
                                    if image_path:
                                        scene['image_path'] = image_path
                                        st.success(f"✅ Regenerated!")
                                        st.rerun()
                    
                    with col_content3:
                        st.write("**Actions:**")
                        
                        if st.button(f"📝 Save Changes", key=f"save_{scene['number']}", use_container_width=True):
                            st.success("✅ Saved!")
                        
                        if has_image and st.button(f"👁️ Preview", key=f"preview_{scene['number']}", use_container_width=True):
                            with st.expander(f"Full Preview - Scene {scene['number']}", expanded=True):
                                st.image(scene['image_path'], use_container_width=True)
                                st.write(f"**Prompt:** {scene['image_prompts'][0] if scene['image_prompts'] else 'No prompt'}")
                        
                        if st.button(f"🗑️ Reset Image", key=f"reset_{scene['number']}", use_container_width=True):
                            if scene.get('image_path'):
                                try:
                                    os.remove(scene['image_path'])
                                except:
                                    pass
                                scene['image_path'] = None
                                st.success("✅ Reset!")
                                st.rerun()
                
                st.divider()
    
    with tab3:
        st.header("🎙️ Audio Generation")
        
        if not st.session_state.scenes:
            st.info("📄 Please import and parse episode files first")
        else:
            if not AUDIO_AVAILABLE:
                st.error("❌ Audio system not available. Please install chatterbox-tts.")
            else:
                # Top controls
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Audio Overview:** {len(st.session_state.scenes)} scenes to process")
                
                with col2:
                    if st.button("🎙️ Generate All Audio", type="primary"):
                        voice_path = getattr(st.session_state, 'voice_file_path', None)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, scene in enumerate(st.session_state.scenes):
                            progress = i / len(st.session_state.scenes)
                            progress_bar.progress(progress)
                            status_text.text(f"Generating audio for scene {scene['number']}")
                            
                            audio_path = st.session_state.generator.generate_scene_audio(
                                scene.get('audio_script', scene['script_text']), scene['number'], voice_path
                            )
                            
                            if audio_path:
                                scene['audio_path'] = audio_path
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ All audio generated!")
                        st.rerun()
                
                with col3:
                    if st.button("🔄 Refresh Audio View"):
                        st.rerun()
                
                st.divider()
                
                # Individual scene audio controls with editable text
                for i, scene in enumerate(st.session_state.scenes):
                    with st.container():
                        # Scene header with audio status
                        col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
                        
                        with col_header1:
                            st.subheader(f"Scene {scene['number']}: {scene['title']}")
                        
                        with col_header2:
                            has_audio = scene.get('audio_path') and os.path.exists(scene.get('audio_path', ''))
                            if has_audio:
                                st.success("🎙️ Audio ✅")
                            else:
                                st.warning("🎙️ No audio")
                        
                        with col_header3:
                            # Audio duration if available
                            if has_audio:
                                try:
                                    import wave
                                    with wave.open(scene['audio_path'], 'r') as wav_file:
                                        frames = wav_file.getnframes()
                                        rate = wav_file.getframerate()
                                        duration = frames / float(rate)
                                        st.info(f"⏱️ {duration:.1f}s")
                                except:
                                    st.info("⏱️ Audio")
                            else:
                                st.info("⏱️ --")
                        
                        # Main content area
                        col_content1, col_content2, col_content3 = st.columns([2, 2, 1])
                        
                        with col_content1:
                            st.write("**Audio Script Text:**")
                            
                            # Editable text for audio generation
                            audio_script = st.text_area(
                                "Audio Script",
                                value=scene['script_text'],
                                height=120,
                                key=f"audio_script_{scene['number']}",
                                help="Edit this text to refine the audio narration. Remove scene markers, add pauses with periods, adjust for natural speech.",
                                label_visibility="collapsed"
                            )
                            
                            # Update scene script if changed
                            if audio_script != scene['script_text']:
                                scene['audio_script'] = audio_script
                            else:
                                scene['audio_script'] = scene['script_text']
                            
                            # Character count and estimated duration
                            char_count = len(audio_script.strip())
                            estimated_duration = char_count / 12  # Rough estimate: ~12 chars per second
                            st.caption(f"📝 {char_count} characters • ~{estimated_duration:.1f}s estimated duration")
                            
                            # Audio generation controls
                            col_gen1, col_gen2 = st.columns(2)
                            
                            with col_gen1:
                                if st.button(f"🎙️ Generate Audio", key=f"gen_audio_{scene['number']}", use_container_width=True):
                                    voice_path = getattr(st.session_state, 'voice_file_path', None)
                                    
                                    with st.spinner(f"Generating audio for scene {scene['number']}..."):
                                        audio_path = st.session_state.generator.generate_scene_audio(
                                            scene.get('audio_script', scene['script_text']), 
                                            scene['number'], 
                                            voice_path
                                        )
                                        
                                        if audio_path:
                                            scene['audio_path'] = audio_path
                                            st.success(f"✅ Audio generated!")
                                            st.rerun()
                            
                            with col_gen2:
                                if st.button(f"🔄 Regenerate", key=f"regen_audio_{scene['number']}", use_container_width=True):
                                    voice_path = getattr(st.session_state, 'voice_file_path', None)
                                    
                                    with st.spinner(f"Regenerating audio for scene {scene['number']}..."):
                                        audio_path = st.session_state.generator.generate_scene_audio(
                                            scene.get('audio_script', scene['script_text']), 
                                            scene['number'], 
                                            voice_path
                                        )
                                        
                                        if audio_path:
                                            scene['audio_path'] = audio_path
                                            st.success(f"✅ Audio regenerated!")
                                            st.rerun()
                        
                        with col_content2:
                            st.write("**Audio Playback:**")
                            
                            if has_audio:
                                st.audio(scene['audio_path'])
                                
                                # Audio file info
                                try:
                                    file_size = os.path.getsize(scene['audio_path']) / 1024  # KB
                                    st.caption(f"📁 {file_size:.1f} KB")
                                except:
                                    pass
                            else:
                                st.info("No audio generated yet")
                            
                            # Audio quality controls
                            st.write("**Audio Settings:**")
                            
                            # Emotion tags for TTS (if supported)
                            emotion = st.selectbox(
                                "Emotion/Style",
                                ["neutral", "excited", "calm", "dramatic", "whisper"],
                                key=f"emotion_{scene['number']}",
                                help="Adjust the emotional tone of the narration"
                            )
                            
                            # Speed control
                            speed = st.slider(
                                "Speech Speed",
                                min_value=0.5,
                                max_value=1.5,
                                value=1.0,
                                step=0.1,
                                key=f"speed_{scene['number']}",
                                help="Adjust speech rate (1.0 = normal)"
                            )
                        
                        with col_content3:
                            st.write("**Actions:**")
                            
                            if st.button(f"📝 Save Script", key=f"save_audio_{scene['number']}", use_container_width=True):
                                scene['script_text'] = scene.get('audio_script', scene['script_text'])
                                st.success("✅ Script saved!")
                            
                            if has_audio and st.button(f"🎵 Preview", key=f"preview_audio_{scene['number']}", use_container_width=True):
                                with st.expander(f"Audio Preview - Scene {scene['number']}", expanded=True):
                                    st.audio(scene['audio_path'])
                                    st.write(f"**Script:** {scene.get('audio_script', scene['script_text'])[:200]}...")
                            
                            if st.button(f"🗑️ Delete Audio", key=f"delete_audio_{scene['number']}", use_container_width=True):
                                if scene.get('audio_path'):
                                    try:
                                        os.remove(scene['audio_path'])
                                    except:
                                        pass
                                    scene['audio_path'] = None
                                    st.success("✅ Audio deleted!")
                                    st.rerun()
                            
                            # Quick text processing buttons
                            if st.button(f"🧹 Clean Text", key=f"clean_{scene['number']}", use_container_width=True):
                                # Remove scene markers and clean text for better TTS
                                clean_text = re.sub(r'\[.*?\]', '', scene['script_text'])
                                clean_text = re.sub(r'\n+', ' ', clean_text)
                                clean_text = clean_text.strip()
                                scene['audio_script'] = clean_text
                                st.success("✅ Text cleaned!")
                                st.rerun()
                    
                    st.divider()
    
    with tab4:
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