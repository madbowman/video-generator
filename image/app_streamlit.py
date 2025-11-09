import streamlit as st
import os
from pathlib import Path
from app_final_timeline import ComfyUIBatchGenerator
import base64
import tempfile

# Page config
st.set_page_config(page_title="Animation Batch Generator", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = ComfyUIBatchGenerator()
    st.session_state.scenes = []
    st.session_state.characters = {}
    st.session_state.generated_images = {}

generator = st.session_state.generator
models = generator.get_available_models()

# Utility functions
def img_to_base64(img_path):
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def build_character_badges(characters_used):
    if not characters_used:
        return ""
    badges = " ".join([f"🏷️ {char.split()[0]}" for char in characters_used])
    return badges

def generate_scene_simple(scene, model_select, scene_num):
    """Generate a scene with character descriptions added to prompt"""
    # Build enhanced prompt with character descriptions
    prompt = scene['prompt']
    
    # Add character descriptions to prompt
    if scene.get('characters_used'):
        char_descriptions = []
        for char_name in scene['characters_used']:
            if char_name in st.session_state.characters:
                desc = st.session_state.characters[char_name]
                # Extract just the key visual traits (first 100 chars)
                brief_desc = desc[:100] if len(desc) > 100 else desc
                char_descriptions.append(f"{char_name}: {brief_desc}")
        
        if char_descriptions:
            char_ref_text = "\n".join(char_descriptions)
            prompt = f"{prompt}\n\nCharacter Appearances:\n{char_ref_text}"
    
    # Create a modified scene with enhanced prompt
    modified_scene = scene.copy()
    modified_scene['prompt'] = prompt
    
    return generator.generate_single(modified_scene, model_select, scene_num)

# Title and tabs
st.title("🎬 Animation Batch Generator")

tab1 = st.tabs(["Timeline"])[0]

with tab1:
    col_settings, col_timeline = st.columns([1, 3])
    
    with col_settings:
        st.subheader("⚙️ Settings")
        
        # File import
        uploaded_file = st.file_uploader("📁 Episode File", type=["txt"])
        if st.button("📥 Import Episode", key="import_btn"):
            if uploaded_file:
                with st.spinner("Parsing episode..."):
                    # Save uploaded file to temp location
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue().decode('utf-8'))
                        tmp_path = tmp.name
                    
                    try:
                        scenes, characters, ok = generator.parse_episode_file(tmp_path)
                        if ok:
                            st.session_state.scenes = scenes
                            st.session_state.characters = characters
                            st.session_state.generated_images = {}
                            st.success(f"✅ Imported {len(scenes)} scenes and {len(characters)} characters")
                        else:
                            st.error("❌ Failed to parse file")
                    finally:
                        os.remove(tmp_path)
            else:
                st.warning("⚠️ Please upload a file first")
        
        st.divider()
        
        # Model selection
        model_select = st.selectbox("🤖 Model", models, index=0)
        
        # Output folder
        output_folder = st.text_input("📂 Output Folder", value=str(generator.output_dir.absolute()))
        
        st.divider()
        
        # Generate buttons
        col_gen_all, col_download = st.columns(2)
        
        with col_gen_all:
            if st.button("🎨 Generate All Scenes", key="gen_all", use_container_width=True):
                if not st.session_state.scenes:
                    st.error("❌ No scenes imported. Please import an episode file first.")
                else:
                    with st.spinner("Generating all scenes..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            if output_folder:
                                generator.output_dir = Path(output_folder)
                                generator.output_dir.mkdir(exist_ok=True)
                            
                            results = []
                            for idx, scene in enumerate(st.session_state.scenes, 1):
                                result = generate_scene_simple(scene, model_select, idx)
                                results.append(result if result else None)
                                if result:
                                    st.session_state.generated_images[idx] = result
                                progress = idx / len(st.session_state.scenes)
                                progress_bar.progress(progress)
                                status_text.text(f"Scene {idx}/{len(st.session_state.scenes)} ✓")
                            
                            st.success(f"✅ Generated {sum(1 for r in results if r)}/{len(results)} scenes")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col_download:
            if st.button("📦 Download as ZIP", key="download", use_container_width=True):
                if not st.session_state.generated_images:
                    st.error("❌ No images to download")
                else:
                    with st.spinner("Creating ZIP..."):
                        try:
                            zip_path = generator.create_zip()
                            with open(zip_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download ZIP",
                                    data=f,
                                    file_name="generated_scenes.zip",
                                    mime="application/zip"
                                )
                        except Exception as e:
                            st.error(f"❌ Error creating ZIP: {str(e)}")
    
    with col_timeline:
        st.subheader("📹 Scenes Timeline")
        
        if not st.session_state.scenes:
            st.info("📂 Import an episode file to begin")
        else:
            # Scene selection dropdown
            scene_titles = [f"Scene {i}: {scene.get('title', f'Scene {i}')}" 
                          for i, scene in enumerate(st.session_state.scenes, 1)]
            selected = st.selectbox("Select a scene", scene_titles, key="scene_select")
            selected_num = int(selected.split(":")[0].split()[-1])
            
            st.divider()
            
            # Display all scenes
            for scene_num, scene in enumerate(st.session_state.scenes, 1):
                with st.container(border=True):
                    # Header
                    col_header_left, col_header_right = st.columns([3, 1])
                    with col_header_left:
                        st.markdown(f"### Scene {scene_num}: {scene['title']}")
                        st.caption(f"Duration: {scene['duration']}")
                    
                    # Main content
                    col_content, col_image = st.columns([2, 1])
                    
                    with col_content:
                        # Prompt
                        st.markdown("**Prompt:**")
                        st.text_area(
                            "Scene prompt",
                            value=scene['prompt'],
                            height=120,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"prompt_{scene_num}"
                        )
                        
                        # Characters
                        if scene.get('characters_used'):
                            st.caption(f"Characters: {build_character_badges(scene['characters_used'])}")
                        
                        # Regenerate button for this scene
                        col_regen, col_spacer = st.columns([1, 2])
                        with col_regen:
                            if st.button("🔄 Regenerate", key=f"regen_{scene_num}", use_container_width=True):
                                with st.spinner(f"Regenerating scene {scene_num}..."):
                                    try:
                                        result = generate_scene_simple(scene, model_select, scene_num)
                                        if result:
                                            st.session_state.generated_images[scene_num] = result
                                            st.success(f"✅ Regenerated scene {scene_num}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to regenerate scene {scene_num}")
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                    
                    with col_image:
                        st.markdown("**Image:**")
                        img_path = st.session_state.generated_images.get(scene_num)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.info("No image generated yet")

# Footer
st.divider()
st.caption("🎬 Animation Batch Generator | Powered by ComfyUI")
