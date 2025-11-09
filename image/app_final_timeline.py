#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import requests
import io
import base64
import zipfile
from pathlib import Path
from PIL import Image
import gradio as gr

class ComfyUIBatchGenerator:
    def __init__(self, comfyui_url="http://127.0.0.1:8000", output_dir="outputs"):
        self.comfyui_url = comfyui_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.character_refs = {}
    
    def create_workflow(self, prompt, model_name, seed, width=1024, height=1024, char_ref_paths=None):
        # The prompt is already enhanced in app_streamlit.py with character descriptions
        # Just use it as-is
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
            "9": {"inputs": {"filename_prefix": "animation_batch", "images": ["8", 0]}, 
                 "class_type": "SaveImage"},
            "11": {"inputs": {"width": width, "height": height, "batch_size": 1}, 
                  "class_type": "EmptyLatentImage"}
        }
    
    def get_available_models(self):
        try:
            response = requests.get(f"{self.comfyui_url}/object_info", timeout=5)
            if response.status_code == 200:
                object_info = response.json()
                if "CheckpointLoaderSimple" in object_info:
                    models = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                    return sorted(models)
        except:
            pass
        return ["disneyPixarCartoon_v10.safetensors"]
    
    def parse_episode_file(self, file_path):
        scenes = []
        characters = {}
        try:
            print(f"=== PARSING FILE: {file_path} ===")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            print(f"File content length: {len(content)} characters")
            print(f"First 200 characters: {content[:200]}")
            
            # Parse characters - look for the CHARACTER REFERENCE section
            lines = content.split('\n')
            i = 0
            in_character_section = False
            
            while i < len(lines):
                line = lines[i].strip()
                
                # More flexible detection of character reference section
                if 'CHARACTER REFERENCE' in line and 'consistency' in line:
                    in_character_section = True
                    print(f"Found character section: {line}")
                    i += 1
                    continue
                
                # Stop parsing characters when we hit scenes or other sections
                if line.startswith('IMAGE') or 'SCENE-BY-SCENE' in line or line.startswith('═══'):
                    in_character_section = False
                
                # Parse character entries in the character section
                if in_character_section and line and ':' in line:
                    # Look for character names like "CASSIAN MIRE (Protagonist, Age 23):"
                    if line.endswith(':') and ('(' in line or line.isupper()):
                        # Extract character name (before parentheses or before colon)
                        if '(' in line:
                            char_name = line.split('(')[0].strip().rstrip(':')
                        else:
                            char_name = line.split(':')[0].strip()
                        
                        if char_name and len(char_name) > 2:  # Valid character name
                            print(f"Found character: {char_name}")
                            # Collect the character description from following lines
                            full_description = line  # Start with the character line
                            j = i + 1
                            while j < len(lines):
                                next_line = lines[j].strip()
                                if not next_line:  # Empty line
                                    break
                                if next_line.startswith('-'):  # Description bullet point
                                    full_description += ' ' + next_line
                                elif ':' in next_line and next_line.endswith(':') and len(next_line.split(':')[0].split()) <= 4:  # Next character
                                    break
                                elif next_line.startswith('IMAGE') or next_line.startswith('SCENE'):
                                    break
                                else:
                                    full_description += ' ' + next_line
                                j += 1
                            
                            characters[char_name] = full_description
                            print(f"Character description: {full_description[:100]}...")
                
                i += 1
            
            # Parse scenes
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('IMAGE') and ':' in line:
                    title = line.split(':', 1)[1].strip()
                    i += 1
                    prompt = []
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('IMAGE'):
                        prompt.append(lines[i].strip())
                        i += 1
                    if prompt:
                        full_prompt = ' '.join(prompt)
                        scenes.append({
                            'title': title, 
                            'prompt': full_prompt, 
                            'duration': '0:04',
                            'characters_used': self.detect_characters(full_prompt, characters.keys())
                        })
                    continue
                i += 1
            
            print(f"Parsing complete. Found {len(scenes)} scenes and {len(characters)} characters")
            print(f"Characters: {list(characters.keys())}")
            
            return scenes, characters, len(scenes) > 0
        except Exception as e:
            print(f"Parse error: {e}")
            return [], {}, False
    
    def detect_characters(self, prompt, character_names):
        """Detect which characters are mentioned in the scene prompt"""
        mentioned = []
        prompt_lower = prompt.lower()
        
        # Sort by length (longest first) to match full names before first names
        sorted_names = sorted(character_names, key=len, reverse=True)
        
        for char_name in sorted_names:
            char_lower = char_name.lower()
            # Check for full name match first (surrounded by word boundaries)
            import re
            if re.search(r'\b' + re.escape(char_lower) + r'\b', prompt_lower):
                mentioned.append(char_name)
            else:
                # Check for first name only if it's longer than 3 chars (avoid false positives)
                first_name = char_name.split()[0].lower()
                if len(first_name) > 3 and re.search(r'\b' + re.escape(first_name) + r'\b', prompt_lower):
                    mentioned.append(char_name)
        
        return mentioned
    
    def generate_single(self, scene, model_name, scene_number, char_ref_paths=None):
        try:
            seed = int(time.time() * 1000) + scene_number
            workflow = self.create_workflow(scene['prompt'], model_name, seed)
            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow})
            prompt_id = response.json()["prompt_id"]
            
            max_wait = 180
            wait_time = 0
            while wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                history_response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history and 'outputs' in history[prompt_id]:
                        break
            
            history = requests.get(f"{self.comfyui_url}/history/{prompt_id}").json()
            if prompt_id in history and 'outputs' in history[prompt_id]:
                for node_output in history[prompt_id]['outputs'].values():
                    if 'images' in node_output:
                        img_info = node_output['images'][0]
                        img_response = requests.get(f"{self.comfyui_url}/view", 
                                                   params={"filename": img_info['filename'], 
                                                          "subfolder": img_info.get('subfolder', ''), 
                                                          "type": "output"})
                        if img_response.status_code == 200:
                            img = Image.open(io.BytesIO(img_response.content))
                            output_path = self.output_dir / f"{scene_number:03d}.png"
                            img.save(output_path)
                            print(f"Saved: {output_path}")
                            return str(output_path)
            return None
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_batch(self, scenes, model_name, progress_callback=None):
        results = []
        for idx, scene in enumerate(scenes):
            if progress_callback:
                progress_callback(idx, len(scenes))
            result = self.generate_single(scene, model_name, idx + 1)
            results.append(result if result else None)
        return results
    
    def generate_character_reference(self, char_name, char_description, model_name):
        """Generate a reference image for a character"""
        try:
            import random
            # Enhanced prompt to ensure single person character reference
            prompt = f"Character reference portrait. SINGLE PERSON. {char_description}. Solo full body standing pose, isolated white background, character design, professional digital art. One person only."
            seed = random.randint(1, 999999)  # Random seed for variation
            
            print(f"\nGenerating character reference for {char_name}")
            print(f"Prompt: {prompt}")
            print(f"Seed: {seed}")
            
            workflow = self.create_workflow(prompt, model_name, seed)
            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow})
            
            if response.status_code != 200:
                print(f"Error: API returned {response.status_code}")
                return None
                
            prompt_id = response.json()["prompt_id"]
            print(f"Started generation with prompt_id: {prompt_id}")
            
            max_wait = 180
            wait_time = 0
            while wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                try:
                    history_response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
                    if history_response.status_code == 200:
                        history = history_response.json()
                        if prompt_id in history and 'outputs' in history[prompt_id]:
                            print(f"Generation completed after {wait_time} seconds")
                            break
                except:
                    pass
            
            history = requests.get(f"{self.comfyui_url}/history/{prompt_id}").json()
            if prompt_id in history and 'outputs' in history[prompt_id]:
                for node_output in history[prompt_id]['outputs'].values():
                    if 'images' in node_output:
                        img_info = node_output['images'][0]
                        img_response = requests.get(f"{self.comfyui_url}/view", 
                                                   params={"filename": img_info['filename'], 
                                                          "subfolder": img_info.get('subfolder', ''), 
                                                          "type": "output"})
                        if img_response.status_code == 200:
                            img = Image.open(io.BytesIO(img_response.content))
                            safe_name = char_name.replace(' ', '_').lower()
                            output_path = self.output_dir / f"char_ref_{safe_name}.png"
                            img.save(output_path)
                            self.character_refs[char_name] = str(output_path)
                            print(f"✅ Saved character ref: {output_path}")
                            return str(output_path)
            
            print(f"❌ No output found in history")
            return None
        except Exception as e:
            print(f"❌ Error generating character reference: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_ui():
    generator = ComfyUIBatchGenerator()
    models = generator.get_available_models()
    app_state = {'scenes': [], 'characters': {}, 'generated_images': {}}
    
    with gr.Blocks(title="Animation Batch Generator", css="""
        .char-badge { display: inline-block; background: #2196F3; color: white; padding: 2px 8px; border-radius: 12px; 
                      font-size: 11px; margin-right: 5px; }
        .gradio-container { padding-bottom: 0 !important; margin-bottom: 0 !important; }
        .gr-block { margin-bottom: 0 !important; }
        body { padding-bottom: 0 !important; }
        .file-preview[style*="display: none"] { display: none !important; }
        .upload-container { display: none !important; }
    """) as app:
        gr.Markdown("# Animation Batch Generator")
        
        with gr.Tabs():
            # TAB 1: Timeline
            with gr.Tab("Timeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")
                        file_input = gr.File(label="Episode File")
                        import_btn = gr.Button("Import Episode", variant="primary")
                        
                        model_select = gr.Dropdown(choices=models, value=models[0], label="Model")
                        output_folder = gr.Textbox(value=str(generator.output_dir.absolute()), label="Output Folder")
                        
                        gr.Markdown("---")
                        gen_all_btn = gr.Button("Generate All Scenes", variant="primary", size="lg")
                        download_all_btn = gr.Button("Download All as ZIP", variant="secondary")
                        download_file = gr.File(label="Download ZIP", visible=False)
                        
                        status = gr.Textbox(label="Status", lines=3)
                        selected_scene = gr.Number(value=0, visible=False)
                        regen_btn = gr.Button("Regen Hidden", visible=False)
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### Scenes Timeline")
                        timeline_container = gr.HTML(value="<p style='text-align: center; padding: 40px; color: #999;'>Import an episode file to begin</p>")
            
            # TAB 2: Characters
            with gr.Tab("Characters"):
                gr.Markdown("### Character Reference Cards")
                gr.Markdown("Characters found in your imported episode file. Generate reference images for consistent character appearance.")
                
                char_container = gr.HTML(value="<p>No characters found. Please import an episode file first.</p>")
        
        # Helper functions
        def img_to_base64(img_path):
            try:
                with open(img_path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except:
                return None
        
        def build_timeline_html(scenes, generated_images):
            html = "<div style='max-height: 75vh; overflow-y: auto;'>"
            
            # Build scene cards
            for i, scene in enumerate(scenes, 1):
                html += build_scene_card(i, scene, generated_images.get(i))
            
            html += """</div>
            <script>
            window.gradioRegenScene = function(sceneNum) {
                const numberInputs = document.querySelectorAll('input[type="number"]');
                if (numberInputs.length >= 2) {
                    numberInputs[1].value = sceneNum;
                    numberInputs[1].dispatchEvent(new Event('input', { bubbles: true }));
                    numberInputs[1].dispatchEvent(new Event('change', { bubbles: true }));
                    setTimeout(() => {
                        const buttons = Array.from(document.querySelectorAll('button'));
                        const regenBtn = buttons.find(b => b.textContent === 'Regen Hidden');
                        if (regenBtn) regenBtn.click();
                    }, 50);
                }
            };
            </script>"""
            return html
        
        def build_scene_card(scene_num, scene, img_path):
            return f"""
            <div style='display: flex; gap: 20px; margin-bottom: 25px; padding: 15px; 
                        background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <div style='flex: 2;'>
                    <div style='font-size: 12px; color: #666; margin-bottom: 5px;'>
                        Scene {scene_num} • {scene['duration']}
                    </div>
                    <div style='font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #000;'>
                        {scene['title']}
                    </div>
                    
                    <div style='margin-bottom: 10px;'>
                        <div style='width: 100%; padding: 10px; border: 1px solid #ddd; 
                                    border-radius: 4px; font-size: 13px; line-height: 1.6; 
                                    background: #f9f9f9; min-height: 120px; color: #000;'>
                            {scene['prompt']}
                        </div>
                    </div>
                    
                    <div style='display: flex; gap: 8px; align-items: center; margin-bottom: 8px;'>
                        <button onclick="window.gradioRegenScene({scene_num})" 
                                style='padding: 8px 16px; background: #2196F3; color: white; border: none; 
                                       border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold;'>
                            Regenerate Scene {scene_num}
                        </button>
                        {build_character_badges(scene.get('characters_used', []))}
                    </div>
                </div>
                <div style='flex: 1; min-width: 280px;'>
                    {build_image_preview(img_path)}
                </div>
            </div>
            """
        
        def build_character_badges(characters_used):
            if not characters_used:
                return ""
            badges = ""
            for char in characters_used:
                ref_status = "✓" if generator.character_refs.get(char) else "○"
                badges += f"<span class='char-badge'>{ref_status} {char.split()[0]}</span>"
            return f"<div style='display: inline-flex; gap: 5px; margin-left: 10px;'>{badges}</div>"
        
        def build_character_cards(characters):
            if not characters:
                return "<p>No characters found. Please import an episode file first.</p>"
            
            cards_html = ""
            
            for char_name, char_description in characters.items():
                cards_html += f"""
                <div class='scene-card' style='display: flex; gap: 20px; margin-bottom: 25px; padding: 15px;
                            background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <div style='flex: 2;'>
                        <h3 style='margin: 0 0 10px 0; color: #333; font-size: 16px;'>{char_name}</h3>
                        <div style='background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #28a745; color: black;'>
                            <strong>Description:</strong> {char_description}
                        </div>
                        <div style='margin-top: 10px;'>
                            <button style='background: #28a745; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;'>
                                🎨 Generate Character Reference
                            </button>
                        </div>
                    </div>
                    <div style='flex: 1; display: flex; justify-content: center; align-items: center;'>
                        <div style='width: 200px; height: 150px; background: #f0f0f0; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #999;'>No Image</div>
                    </div>
                </div>
                """
            
            return cards_html
        
        def build_image_preview(img_path):
            if img_path and os.path.exists(img_path):
                img_base64 = img_to_base64(img_path)
                if img_base64:
                    return f"""
                    <img src='data:image/png;base64,{img_base64}' 
                         style='width: 100%; border-radius: 8px; border: 2px solid #4CAF50;'>
                    <div style='margin-top: 5px; font-size: 11px; color: #4CAF50; text-align: center;'>
                        ✓ {os.path.basename(img_path)}
                    </div>
                    """
            return """
            <div style='width: 100%; aspect-ratio: 1; background: #f5f5f5; 
                        border-radius: 8px; border: 2px dashed #ccc; display: flex; 
                        align-items: center; justify-content: center; color: #999;'>
                Not generated
            </div>
            """
        
        def import_file(file):
            if not file:
                return (
                    "<p style='text-align: center; padding: 40px; color: #999;'>Import an episode file to begin</p>",
                    "Error: No file",
                    "<p>No characters found. Please import an episode file first.</p>"
                )
            
            scenes, characters, ok = generator.parse_episode_file(file.name)
            if ok:
                app_state['scenes'] = scenes
                app_state['characters'] = characters
                app_state['generated_images'] = {}
                timeline_html = build_timeline_html(scenes, app_state['generated_images'])
                char_html = build_character_cards(characters)
                
                return (
                    timeline_html,
                    f"Imported {len(scenes)} scenes and {len(characters)} characters",
                    char_html
                )
            
            return (
                "<p style='text-align: center; padding: 40px; color: #999;'>Import an episode file to begin</p>",
                "Error importing file",
                "<p>No characters found. Please import an episode file first.</p>"
            )
        
        def generate_all(model, output_path, progress=gr.Progress()):
            if not app_state['scenes']:
                return timeline_container.value, "No scenes to generate"
            
            if output_path:
                generator.output_dir = Path(output_path)
                generator.output_dir.mkdir(exist_ok=True)
            
            def cb(curr, total):
                progress((curr+1)/total, f"Scene {curr+1}/{total}")
            
            results = generator.generate_batch(app_state['scenes'], model, cb)
            
            for i, result in enumerate(results, 1):
                if result:
                    app_state['generated_images'][i] = result
            
            html = build_timeline_html(app_state['scenes'], app_state['generated_images'])
            success_count = sum(1 for r in results if r)
            return html, f"Generated {success_count}/{len(results)} scenes"
        
        def add_character_reference(char_name, char_image):
            if not char_name or not char_image:
                return "Enter name and upload image", "<p>No characters uploaded yet</p>"
            
            generator.character_refs[char_name] = char_image
            
            html = "<div>"
            for name, img_path in generator.character_refs.items():
                img_base64 = img_to_base64(img_path)
                if img_base64:
                    html += f"""
                    <div style='margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 8px;'>
                        <div style='font-weight: bold; margin-bottom: 5px;'>{name}</div>
                        <img src='data:image/png;base64,{img_base64}' style='width: 200px; border-radius: 4px;'>
                    </div>
                    """
            html += "</div>"
            
            return f"Added reference for {char_name}", html
        
        def regenerate_scene(scene_num):
            """Regenerate a single scene"""
            scene_num = int(scene_num)
            
            if not app_state['scenes'] or scene_num < 1 or scene_num > len(app_state['scenes']):
                return "<p>No scenes</p>", f"❌ Invalid scene number: {scene_num}"
            
            try:
                scene = app_state['scenes'][scene_num - 1]
                model = models[0]
                
                result = generator.generate_single_scene(scene, model)
                
                if result:
                    app_state['generated_images'][scene_num] = result
                    html = build_timeline_html(app_state['scenes'], app_state['generated_images'])
                    return html, f"✅ Regenerated scene {scene_num}: {scene['title']}"
                else:
                    return build_timeline_html(app_state['scenes'], app_state['generated_images']), f"❌ Failed to regenerate scene {scene_num}"
                    
            except Exception as e:
                return build_timeline_html(app_state['scenes'], app_state['generated_images']), f"❌ Error: {str(e)}"
        
        def create_zip():
            if not app_state['generated_images']:
                return None, "No images to download"
            
            zip_path = generator.output_dir / "all_scenes.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for scene_num, img_path in sorted(app_state['generated_images'].items()):
                    if os.path.exists(img_path):
                        zipf.write(img_path, os.path.basename(img_path))
            
            return str(zip_path), f"Created ZIP with {len(app_state['generated_images'])} images"
        
        def regenerate_single_scene(scene_num, updated_prompt, model):
            """Regenerate a single scene with updated prompt"""
            if not app_state['scenes'] or not (1 <= scene_num <= len(app_state['scenes'])):
                return timeline_container.value, f"❌ Invalid scene number: {scene_num}"
            
            try:
                scene_index = scene_num - 1
                scene = app_state['scenes'][scene_index].copy()
                
                # Update the prompt if provided
                if updated_prompt and updated_prompt.strip():
                    scene['prompt'] = updated_prompt.strip()
                    # Also update in app_state so it persists
                    app_state['scenes'][scene_index]['prompt'] = updated_prompt.strip()
                
                result = generator.generate_single(scene, model, scene_num)
                
                if result:
                    app_state['generated_images'][scene_num] = result
                    html = build_timeline_html(app_state['scenes'], app_state['generated_images'])
                    return html, f"✅ Regenerated scene {scene_num}: {scene['title']}"
                else:
                    return timeline_container.value, f"❌ Failed to regenerate scene {scene_num}"
                    
            except Exception as e:
                return timeline_container.value, f"❌ Error: {str(e)}"
        
        # Event handlers
        import_btn.click(import_file, [file_input], [timeline_container, status, char_container])
        gen_all_btn.click(generate_all, [model_select, output_folder], [timeline_container, status])
        download_all_btn.click(create_zip, outputs=[download_file, status])
        regen_btn.click(regenerate_scene, [selected_scene], [timeline_container, status])
    
    return app

if __name__ == "__main__":
    create_ui().launch(server_name="127.0.0.1", server_port=7860)
