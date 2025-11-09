"""
Chatterbox TTS - Complete GUI (REFACTORED)
Fixed: Status tracking, non-functional parameters, progress bars, FFmpeg issues
"""

import gradio as gr
import torch
import json
import os
import re
import time
from pathlib import Path
import numpy as np
from scipy.io import wavfile

# Wrapper to avoid FFmpeg/TorchCodec issues
class ta:
    @staticmethod
    def save(filepath, waveform, sample_rate):
        """Save audio using scipy instead of torchaudio"""
        audio_np = waveform.cpu().squeeze().numpy() if isinstance(waveform, torch.Tensor) else waveform
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(filepath, sample_rate, audio_int16)
    
    @staticmethod
    def load(filepath):
        """Load audio using scipy"""
        sample_rate, audio_np = wavfile.read(filepath)
        audio_tensor = torch.from_numpy(audio_np.astype(np.float32) / 32767.0)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        return audio_tensor, sample_rate

from chatterbox_story_tts import StoryNarrator

# Try to import voice conversion
try:
    from chatterbox.vc import ChatterboxVC
    VOICE_CONVERSION_AVAILABLE = True
except:
    VOICE_CONVERSION_AVAILABLE = False

# Try multilingual
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    MULTILINGUAL_AVAILABLE = True
except:
    MULTILINGUAL_AVAILABLE = False


class ChatterboxGUI:
    """Complete Chatterbox GUI with all features - REFACTORED"""
    
    def __init__(self):
        self.narrator = None
        self.vc_model = None
        self.multi_model = None
        
        # Auto-detect CUDA/GPU
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"🎮 GPU DETECTED: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = "cpu"
            print("🖥️  GPU not available, using CPU")
        
        # CPU optimizations (even for GPU, helps with preprocessing)
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        logical_count = psutil.cpu_count(logical=True)
        
        # Set environment variables for maximum CPU utilization
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
        
        torch.set_num_threads(cpu_count)
        print(f"   🚀 Using ALL {cpu_count} CPU cores ({logical_count} threads)")
        
        # Create output directory
        self.output_dir = Path("chatterbox_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎙️ Chatterbox GUI initialized on {self.device}")
    
    def basic_tts(self, text, voice_file, exaggeration, cfg_weight, temperature, pause_ms, progress=gr.Progress()):
        """Basic TTS generation - strips all [] tags"""
        try:
            # Validate inputs
            if not text or not text.strip():
                return None, "❌ Please enter some text"
            
            # Load narrator
            progress(0, desc="Loading TTS model...")
            if self.narrator is None or voice_file:
                self.narrator = StoryNarrator(
                    voice_sample_path=voice_file if voice_file else None,
                    device=self.device
                )
            
            # Clean text - remove all [] tags
            progress(0.1, desc="Processing text...")
            cleaned_text = self.narrator.clean_text_for_basic_tts(text)
            
            if not cleaned_text.strip():
                return None, "❌ No text remaining after removing tags"
            
            # Count sentences for progress
            sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
            sentence_count = len(sentences)
            
            progress(0.2, desc=f"Generating {sentence_count} sentences...")
            
            output_path = self.output_dir / "basic_output.wav"
            start_time = time.time()
            
            # Generate with progress tracking
            def progress_update(percent, eta, status):
                # Extract simple progress info
                desc = f"{int(percent)}% | ETA: {eta}"
                progress(percent / 100, desc=desc)
            
            self.narrator.narrate_story(
                story_text=cleaned_text,
                output_path=str(output_path),
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                pause_between_sentences_ms=int(pause_ms),
                progress_callback=progress_update
            )
            
            elapsed = time.time() - start_time
            progress(1.0, desc="Complete!")
            
            status = f"✅ Generated: {output_path.name}\n"
            status += f"⏱️ Time: {elapsed/60:.1f}m ({sentence_count} sentences)\n"
            status += f"📊 Speed: {sentence_count/(elapsed/60):.1f} sentences/min"
            
            return str(output_path), status
        
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def emotion_tagged_tts(self, text, voice_file, emotion_json, default_exag, default_cfg, default_temp, pause_ms, progress=gr.Progress()):
        """TTS with emotion tags - supports both [emotion] tags and JSON"""
        try:
            # Validate inputs
            if not text or not text.strip():
                return None, "❌ Please enter some text"
            
            # Load narrator
            progress(0, desc="Loading TTS model...")
            if self.narrator is None or voice_file:
                self.narrator = StoryNarrator(
                    voice_sample_path=voice_file if voice_file else None,
                    device=self.device
                )
            
            # Check if text has [emotion] tags
            if re.search(r'\[\w+\]', text):
                progress(0.1, desc="Parsing emotion tags...")
                # Parse emotion tags from text
                parsed_sentences = self.narrator.parse_emotion_tags(text)
                
                # Generate audio segments with emotions
                audio_segments = []
                total_sentences = len(parsed_sentences)
                
                for i, (sentence, emotion_params) in enumerate(parsed_sentences):
                    if not sentence.strip():
                        continue
                    
                    # Update progress
                    current_progress = 0.2 + (i / total_sentences) * 0.75
                    desc = f"Sentence {i + 1}/{total_sentences}"
                    progress(current_progress, desc=desc)
                    
                    # Use emotion parameters if available
                    exag = emotion_params.get('exaggeration', default_exag)
                    cfg = emotion_params.get('cfg_weight', default_cfg)
                    temp = emotion_params.get('temperature', default_temp)
                    
                    wav = self.narrator.generate_sentence(sentence, exag, cfg, temp)
                    audio_segments.append(wav)
                    
                    # Add pause
                    if i < len(parsed_sentences) - 1:
                        pause = self.narrator.add_pause(int(pause_ms))
                        audio_segments.append(pause)
                
                # Concatenate
                progress(0.95, desc="Finalizing audio...")
                if audio_segments:
                    full_audio = torch.cat(audio_segments)
                    output_path = self.output_dir / "emotion_output.wav"
                    
                    if full_audio.dim() == 1:
                        full_audio = full_audio.unsqueeze(0)
                    
                    ta.save(str(output_path), full_audio, self.narrator.sample_rate)
                    progress(1.0, desc="Complete!")
                    
                    status = f"✅ Generated with emotion tags: {output_path.name}\n"
                    status += f"📊 Processed {total_sentences} sentences with emotions"
                    
                    return str(output_path), status
                else:
                    return None, "❌ No valid sentences found"
                    
            else:
                # Use JSON emotion tags
                emotion_tags = None
                if emotion_json.strip():
                    try:
                        progress(0.1, desc="Parsing JSON emotion tags...")
                        emotion_tags = json.loads(emotion_json)
                    except:
                        return None, "❌ Invalid JSON format"
                
                progress(0.2, desc="Generating with emotions...")
                output_path = self.output_dir / "emotion_output.wav"
                
                # Generate with progress tracking
                def progress_update(percent, eta, status):
                    desc = f"{int(percent)}% | ETA: {eta}"
                    progress(0.2 + (percent / 100) * 0.75, desc=desc)
                
                self.narrator.narrate_story(
                    story_text=text,
                    output_path=str(output_path),
                    exaggeration=default_exag,
                    cfg_weight=default_cfg,
                    temperature=default_temp,
                    pause_between_sentences_ms=int(pause_ms),
                    emotion_tags=emotion_tags,
                    progress_callback=progress_update
                )
                
                progress(1.0, desc="Complete!")
                
                status = f"✅ Generated with JSON emotions: {output_path.name}"
                return str(output_path), status
        
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def multi_speaker_tts(self, dialog_text, 
                          speaker_1_voice, speaker_1_name,
                          speaker_2_voice, speaker_2_name,
                          speaker_3_voice, speaker_3_name,
                          speaker_4_voice, speaker_4_name,
                          speaker_5_voice, speaker_5_name,
                          speaker_6_voice, speaker_6_name,
                          exaggeration, cfg_weight, temperature, pause_ms, progress=gr.Progress()):
        """Multi-speaker dialog generation"""
        try:
            # Validate inputs
            if not dialog_text or not dialog_text.strip():
                return None, "❌ Please enter dialog text"
            
            progress(0, desc="Parsing dialog...")
            
            if self.narrator is None:
                self.narrator = StoryNarrator(device=self.device)
            
            # Parse dialog
            parsed_lines = self.narrator.parse_multi_speaker_tags(dialog_text)
            
            if not parsed_lines:
                return None, "❌ No dialog found! Use format: [speaker]text or [speaker,emotion]text"
            
            # Build speaker mapping from individual parameters
            speaker_mapping = {}
            speakers_data = [
                (speaker_1_voice, speaker_1_name),
                (speaker_2_voice, speaker_2_name),
                (speaker_3_voice, speaker_3_name),
                (speaker_4_voice, speaker_4_name),
                (speaker_5_voice, speaker_5_name),
                (speaker_6_voice, speaker_6_name),
            ]
            
            for i, (voice_file, name) in enumerate(speakers_data):
                if name and name.strip() and voice_file:
                    speaker_mapping[name.strip().lower()] = voice_file
                elif voice_file:
                    # Generic fallback
                    speaker_mapping[str(i + 1)] = voice_file
            
            if not speaker_mapping:
                return None, "❌ Please upload at least one voice file with a name"
            
            # Generate dialog
            audio_segments = []
            total_lines = len(parsed_lines)
            start_time = time.time()
            
            for idx, (speaker, content, emotion_params) in enumerate(parsed_lines):
                if not content.strip():
                    continue
                
                speaker_lower = speaker.strip().lower()
                
                # Get voice file
                voice_file = speaker_mapping.get(speaker_lower)
                if not voice_file:
                    # Try numeric fallback
                    voice_file = speaker_mapping.get('1')  # Default to first speaker
                
                if not voice_file:
                    continue
                
                # Update progress
                current_progress = (idx + 1) / total_lines
                elapsed = time.time() - start_time
                if idx > 0:
                    eta_seconds = (elapsed / (idx + 1)) * (total_lines - (idx + 1))
                    eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                else:
                    eta_str = "calculating..."
                
                desc = f"Line {idx + 1}/{total_lines} | ETA: {eta_str}"
                progress(current_progress, desc=desc)
                
                # Get emotion parameters
                current_exag = emotion_params.get('exaggeration', exaggeration)
                current_cfg = emotion_params.get('cfg_weight', cfg_weight)
                current_temp = emotion_params.get('temperature', temperature)
                
                # Create narrator with this voice
                current_narrator = StoryNarrator(
                    voice_sample_path=voice_file,
                    device=self.device
                )
                
                # Generate audio
                wav = current_narrator.generate_sentence(
                    content,
                    exaggeration=current_exag,
                    cfg_weight=current_cfg,
                    temperature=current_temp
                )
                
                audio_segments.append(wav)
                
                # Add pause
                if idx < len(parsed_lines) - 1:
                    speaker_pause = current_narrator.add_pause(int(pause_ms))
                    audio_segments.append(speaker_pause)
            
            # Finalize
            progress(0.95, desc="Finalizing audio...")
            
            if audio_segments and len(audio_segments) > 1:
                audio_segments = audio_segments[:-1]  # Remove last pause
            
            if audio_segments:
                final_audio = torch.cat(audio_segments)
                output_path = self.output_dir / "multi_speaker_output.wav"
                
                if final_audio.dim() == 1:
                    final_audio = final_audio.unsqueeze(0)
                
                ta.save(str(output_path), final_audio, self.narrator.sample_rate)
                
                elapsed_total = time.time() - start_time
                progress(1.0, desc="Complete!")
                
                status = f"✅ Multi-speaker dialog: {output_path.name}\n"
                status += f"📊 {len(audio_segments)} segments from {total_lines} lines\n"
                status += f"⏱️ Time: {elapsed_total/60:.1f}m"
                
                return str(output_path), status
            else:
                return None, "❌ No valid dialog content found"
        
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def voice_conversion(self, source_audio, target_voice, progress=gr.Progress()):
        """Convert voice in audio to target voice"""
        try:
            if not VOICE_CONVERSION_AVAILABLE:
                return None, "❌ Voice conversion not available. Install full chatterbox package."
            
            if not source_audio or not target_voice:
                return None, "❌ Please provide both source audio and target voice"
            
            progress(0, desc="Loading voice conversion model...")
            
            if self.vc_model is None:
                from chatterbox.vc import ChatterboxVC
                self.vc_model = ChatterboxVC.from_pretrained(device=self.device)
            
            progress(0.5, desc="Converting voice...")
            
            # Use correct parameter names: audio and target_voice_path
            converted = self.vc_model.generate(
                audio=source_audio,
                target_voice_path=target_voice
            )
            
            progress(0.9, desc="Saving converted audio...")
            
            # Save
            output_path = self.output_dir / "voice_converted.wav"
            ta.save(str(output_path), converted.unsqueeze(0) if converted.dim() == 1 else converted, self.vc_model.sr)
            
            progress(1.0, desc="Complete!")
            
            return str(output_path), f"✅ Voice converted: {output_path.name}"
        
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def multilingual_tts(self, text, language, voice_file, exaggeration, cfg_weight, temperature, progress=gr.Progress()):
        """Multilingual TTS (23 languages)"""
        try:
            if not MULTILINGUAL_AVAILABLE:
                return None, "❌ Multilingual not available. Install: pip install chatterbox-tts[multilingual]"
            
            if not text or not text.strip():
                return None, "❌ Please enter some text"
            
            progress(0, desc="Loading multilingual model...")
            
            if self.multi_model is None:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.multi_model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            
            progress(0.4, desc=f"Generating in {language}...")
            
            # Generate
            if voice_file:
                wav = self.multi_model.generate(
                    text,
                    language_id=language,
                    audio_prompt_path=voice_file,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
            else:
                wav = self.multi_model.generate(
                    text,
                    language_id=language,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
            
            progress(0.9, desc="Saving audio...")
            
            # Save
            output_path = self.output_dir / f"multilingual_{language}.wav"
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            ta.save(str(output_path), wav, self.multi_model.sr)
            
            progress(1.0, desc="Complete!")
            
            return str(output_path), f"✅ Generated in {language}: {output_path.name}"
        
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg


def create_gui():
    """Create the Gradio interface"""
    
    app = ChatterboxGUI()
    
    # Language options
    languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", 
                 "ar", "hi", "da", "nl", "fi", "el", "he", "ms", "no", "pl", 
                 "sv", "sw", "tr"]
    
    with gr.Blocks(title="Chatterbox TTS Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎙️ Chatterbox TTS Studio (REFACTORED)
        ### Complete voice synthesis suite with improved status tracking
        """)
        
        # System info
        with gr.Row():
            gr.Markdown(f"""
            **Device:** {app.device.upper()} | 
            **Voice Conversion:** {"✅" if VOICE_CONVERSION_AVAILABLE else "❌"} | 
            **Multilingual:** {"✅" if MULTILINGUAL_AVAILABLE else "❌"}
            """)
        
        with gr.Tabs():
            # Tab 1: Basic TTS
            with gr.Tab("🎵 Basic TTS"):
                gr.Markdown("### Simple text-to-speech (ignores all [tags])")
                
                with gr.Row():
                    with gr.Column():
                        basic_text = gr.Textbox(
                            label="Text to speak",
                            placeholder="Enter your text... (all [tags] will be ignored)",
                            lines=8
                        )
                        basic_voice = gr.Audio(
                            label="Voice sample (optional - for cloning)",
                            type="filepath"
                        )
                        
                        with gr.Row():
                            basic_exag = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Exaggeration (emotion intensity)")
                            basic_cfg = gr.Slider(0.2, 1.0, value=0.4, step=0.05, label="CFG Weight (speed control)")
                        
                        with gr.Row():
                            basic_temp = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature (variation)")
                            basic_pause = gr.Slider(0, 2000, value=500, step=100, label="Pause between sentences (ms)")
                        
                        basic_btn = gr.Button("🎵 Generate", variant="primary", size="lg")
                    
                    with gr.Column():
                        basic_audio_out = gr.Audio(
                            label="Generated Audio", 
                            type="filepath",
                            show_download_button=True
                        )
                        basic_status = gr.Textbox(
                            label="Status", 
                            lines=5,
                            interactive=False
                        )
                
                basic_btn.click(
                    app.basic_tts,
                    inputs=[basic_text, basic_voice, basic_exag, basic_cfg, basic_temp, basic_pause],
                    outputs=[basic_audio_out, basic_status]
                )
            
            # Tab 2: Emotion Tags
            with gr.Tab("🎭 Emotion Tags"):
                gr.Markdown("### Per-sentence emotion control")
                
                with gr.Row():
                    with gr.Column():
                        emotion_text = gr.Textbox(
                            label="Story text with emotion tags",
                            placeholder="[happy]Great news!\n[sad]But then...",
                            lines=10
                        )
                        emotion_voice = gr.Audio(
                            label="Voice sample (optional)",
                            type="filepath"
                        )
                        emotion_json = gr.Code(
                            label="JSON Emotion Tags (legacy)",
                            value='{\n  "0": {"exaggeration": 1.5, "cfg_weight": 0.3}\n}',
                            language="json",
                            lines=4
                        )
                        
                        with gr.Row():
                            emotion_exag = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Default Exaggeration")
                            emotion_cfg = gr.Slider(0.2, 1.0, value=0.4, step=0.05, label="Default CFG")
                        
                        with gr.Row():
                            emotion_temp = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Default Temperature")
                            emotion_pause = gr.Slider(0, 2000, value=500, step=100, label="Pause (ms)")
                        
                        emotion_btn = gr.Button("🎭 Generate with Emotions", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Available Emotions:**
                        - `happy`, `sad`, `angry`, `excited`
                        - `tired`, `scared`, `calm`, `dramatic`
                        - `whisper`, `shouting`, `confused`
                        - `surprised`, `concerned`
                        
                        **Usage:**
                        ```
                        [happy]Hello there!
                        [sad]I'm feeling down.
                        [angry]This is wrong!
                        ```
                        """)
                        emotion_audio_out = gr.Audio(
                            label="Generated Audio",
                            type="filepath",
                            show_download_button=True
                        )
                        emotion_status = gr.Textbox(
                            label="Status",
                            lines=5,
                            interactive=False
                        )
                
                emotion_btn.click(
                    app.emotion_tagged_tts,
                    inputs=[emotion_text, emotion_voice, emotion_json, emotion_exag, emotion_cfg, emotion_temp, emotion_pause],
                    outputs=[emotion_audio_out, emotion_status]
                )
            
            # Tab 3: Multi-Speaker
            with gr.Tab("👥 Multi-Speaker Dialog"):
                gr.Markdown("### Create conversations with different voices")
                
                with gr.Row():
                    with gr.Column():
                        dialog_text = gr.Textbox(
                            label="Dialog (use [speaker,emotion] format)",
                            placeholder="[narrator]The story begins.\n[hero,happy]Hello!\n[villain,angry]No!",
                            lines=15
                        )
                        
                        gr.Markdown("### Speaker Setup")
                        
                        speakers = []
                        for i in range(1, 7):
                            with gr.Row():
                                voice = gr.Audio(label=f"Speaker {i} Voice", type="filepath")
                                name = gr.Textbox(label=f"Speaker {i} Name", placeholder=f"{i}")
                                speakers.append((voice, name))
                        
                        with gr.Row():
                            dialog_exag = gr.Slider(0.25, 2.0, value=0.6, step=0.05, label="Default Exaggeration")
                            dialog_cfg = gr.Slider(0.2, 1.0, value=0.4, step=0.05, label="Default CFG")
                        
                        with gr.Row():
                            dialog_temp = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Default Temperature")
                            dialog_pause = gr.Slider(0, 2000, value=500, step=100, label="Pause (ms)")
                        
                        dialog_btn = gr.Button("👥 Generate Dialog", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Format:** `[speakername,emotion]text`
                        
                        ```
                        [narrator]The story begins.
                        [hero,happy]Great day!
                        [villain,angry]Not for long!
                        [narrator]The battle starts.
                        ```
                        
                        **Setup:**
                        1. Upload voices for speakers
                        2. Enter names (or use 1,2,3...)
                        3. Use those names in [brackets]
                        """)
                        dialog_audio_out = gr.Audio(
                            label="Generated Dialog",
                            type="filepath",
                            show_download_button=True
                        )
                        dialog_status = gr.Textbox(
                            label="Status",
                            lines=5,
                            interactive=False
                        )
                
                # Unpack all speaker inputs individually
                all_inputs = [dialog_text]
                for voice, name in speakers:
                    all_inputs.extend([voice, name])
                all_inputs.extend([dialog_exag, dialog_cfg, dialog_temp, dialog_pause])
                
                dialog_btn.click(
                    app.multi_speaker_tts,
                    inputs=all_inputs,
                    outputs=[dialog_audio_out, dialog_status]
                )
            
            # Tab 4: Voice Conversion
            with gr.Tab("🗣️ Voice Conversion"):
                gr.Markdown("### Convert any voice to another")
                
                with gr.Row():
                    with gr.Column():
                        vc_source = gr.Audio(
                            label="Source audio (voice to convert)",
                            type="filepath"
                        )
                        vc_target = gr.Audio(
                            label="Target voice (voice to sound like)",
                            type="filepath"
                        )
                        vc_btn = gr.Button("🗣️ Convert Voice", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("""
                        **How it works:**
                        1. Upload audio with original voice
                        2. Upload target voice sample (10-20s)
                        3. Get speech in new voice!
                        
                        ⚠️ Requires voice conversion model
                        """)
                        vc_audio_out = gr.Audio(
                            label="Converted Audio",
                            type="filepath",
                            show_download_button=True
                        )
                        vc_status = gr.Textbox(
                            label="Status",
                            lines=3,
                            interactive=False
                        )
                
                vc_btn.click(
                    app.voice_conversion,
                    inputs=[vc_source, vc_target],
                    outputs=[vc_audio_out, vc_status]
                )
            
            # Tab 5: Multilingual
            with gr.Tab("🌍 Multilingual"):
                gr.Markdown("### Generate speech in 23 languages")
                
                with gr.Row():
                    with gr.Column():
                        multi_text = gr.Textbox(
                            label="Text to speak",
                            placeholder="Enter text in target language...",
                            lines=6
                        )
                        multi_lang = gr.Dropdown(
                            choices=languages,
                            value="en",
                            label="Language"
                        )
                        multi_voice = gr.Audio(
                            label="Voice sample (optional)",
                            type="filepath"
                        )
                        
                        with gr.Row():
                            multi_exag = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Exaggeration")
                            multi_cfg = gr.Slider(0.2, 1.0, value=0.4, step=0.05, label="CFG Weight")
                        
                        multi_temp = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                        multi_btn = gr.Button("🌍 Generate", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Supported Languages:**
                        Arabic, Chinese, Danish, Dutch, English,
                        Finnish, French, German, Greek, Hebrew,
                        Hindi, Italian, Japanese, Korean, Malay,
                        Norwegian, Polish, Portuguese, Russian,
                        Spanish, Swedish, Swahili, Turkish
                        
                        ⚠️ Requires multilingual model
                        """)
                        multi_audio_out = gr.Audio(
                            label="Generated Audio",
                            type="filepath",
                            show_download_button=True
                        )
                        multi_status = gr.Textbox(
                            label="Status",
                            lines=3,
                            interactive=False
                        )
                
                multi_btn.click(
                    app.multilingual_tts,
                    inputs=[multi_text, multi_lang, multi_voice, multi_exag, multi_cfg, multi_temp],
                    outputs=[multi_audio_out, multi_status]
                )
        
        gr.Markdown("""
        ---
        **Output Directory:** `chatterbox_outputs/` | **Device:** CPU-optimized
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Chatterbox TTS Studio (Refactored)...")
    print("📂 Output directory: chatterbox_outputs/")
    
    demo = create_gui()
    print("\n🌐 Server starting at: http://127.0.0.1:7862/")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )
