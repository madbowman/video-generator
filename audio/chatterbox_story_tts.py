"""
Chatterbox TTS - Story Narrator (FIXED VERSION)
Fixes dimension mismatch errors
"""

import re
import torch
from pathlib import Path
import time
import psutil
import numpy as np
from scipy.io import wavfile

# Wrapper to replace torchaudio.save (avoids FFmpeg/TorchCodec issues)
class ta:
    @staticmethod
    def save(filepath, waveform, sample_rate):
        """Save audio using scipy instead of torchaudio to avoid FFmpeg issues"""
        audio_np = waveform.cpu().squeeze().numpy() if isinstance(waveform, torch.Tensor) else waveform
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(filepath, sample_rate, audio_int16)

# Try to import chatterbox
try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    print("âš ï¸  Chatterbox not installed. Run: pip install chatterbox-tts")

# Try to import CPU optimizer
try:
    from cpu_optimizer import optimize_cpu_performance
    CPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPU_OPTIMIZER_AVAILABLE = False

# Emotion mapping for named emotions
EMOTION_MAPPING = {
    'happy': {'exaggeration': 0.8, 'cfg_weight': 0.45, 'temperature': 0.7},
    'sad': {'exaggeration': 0.4, 'cfg_weight': 0.3, 'temperature': 0.5},
    'angry': {'exaggeration': 1.2, 'cfg_weight': 0.4, 'temperature': 0.8},
    'excited': {'exaggeration': 1.0, 'cfg_weight': 0.6, 'temperature': 0.8},
    'tired': {'exaggeration': 0.3, 'cfg_weight': 0.25, 'temperature': 0.4},
    'scared': {'exaggeration': 1.1, 'cfg_weight': 0.35, 'temperature': 0.6},
    'calm': {'exaggeration': 0.4, 'cfg_weight': 0.4, 'temperature': 0.5},
    'dramatic': {'exaggeration': 1.5, 'cfg_weight': 0.3, 'temperature': 0.7},
    'whisper': {'exaggeration': 0.2, 'cfg_weight': 0.2, 'temperature': 0.3},
    'shouting': {'exaggeration': 1.8, 'cfg_weight': 0.7, 'temperature': 0.9},
    'confused': {'exaggeration': 0.6, 'cfg_weight': 0.4, 'temperature': 0.6},
    'surprised': {'exaggeration': 1.0, 'cfg_weight': 0.5, 'temperature': 0.8},
    'concerned': {'exaggeration': 0.6, 'cfg_weight': 0.35, 'temperature': 0.5}
}


class StoryNarrator:
    """
    Handles long-form story narration with Chatterbox TTS
    Processes sentence-by-sentence and concatenates seamlessly
    """
    
    def __init__(self, voice_sample_path=None, device="cpu"):
        """
        Initialize the narrator
        
        Args:
            voice_sample_path: Path to reference voice audio (optional)
            device: "cpu" for CPU (default), "cuda" for GPU
        """
        if not CHATTERBOX_AVAILABLE:
            raise ImportError("Chatterbox TTS not installed")
        
        # Force CPU for stability and compatibility
        if device == "cuda":
            print("âš ï¸  GPU mode disabled for stability. Using CPU for optimal performance.")
            device = "cpu"
        
        # Apply CPU optimizations
        if CPU_OPTIMIZER_AVAILABLE:
            self.aggressive_memory = optimize_cpu_performance()
        else:
            # Fallback optimization - use ALL cores for maximum speed
            import psutil
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            logical_count = psutil.cpu_count(logical=True)  # Logical cores
            torch.set_num_threads(cpu_count)  # Use all physical cores
            self.aggressive_memory = False
            print(f"   Using ALL {cpu_count} CPU cores ({logical_count} threads)")
        
        print(f"ðŸŽ™ï¸  Loading Chatterbox model on {device} with CPU optimizations...")
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.voice_sample = voice_sample_path
        self.device = device
        self.sample_rate = self.model.sr
        
        print(f"âœ… Model loaded! Sample rate: {self.sample_rate} Hz")
    
    def split_into_sentences(self, text):
        """
        Split text into sentences intelligently
        Handles common abbreviations and edge cases
        """
        # Handle common abbreviations that shouldn't split
        text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs").replace("Ms.", "Ms")
        text = text.replace("Dr.", "Dr").replace("Prof.", "Prof")
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _ensure_1d(self, tensor):
        """Ensure tensor is 1D by squeezing extra dimensions"""
        while tensor.dim() > 1:
            tensor = tensor.squeeze(0)
        return tensor
    
    def generate_sentence(self, text, exaggeration=0.5, cfg_weight=0.5, temperature=0.7):
        """
        Generate audio for a single sentence with emotion control
        
        Args:
            text: The text to speak
            exaggeration: 0.25-2.0, controls emotional intensity (higher = more dramatic)
            cfg_weight: 0.2-1.0, controls pacing (lower = slower, more deliberate)
            temperature: 0.0-1.0, affects expressiveness
        
        Returns:
            torch.Tensor: Audio waveform (1D)
        """
        print(f"  ðŸŽµ Generating: '{text[:50]}...'")
        
        # Generate with emotion parameters
        if self.voice_sample:
            wav = self.model.generate(
                text,
                audio_prompt_path=self.voice_sample,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        else:
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        # Ensure 1D tensor - this is critical!
        wav = self._ensure_1d(wav)
        
        return wav
    
    def add_pause(self, duration_ms=500):
        """
        Create silence for pauses between sentences
        
        Args:
            duration_ms: Pause duration in milliseconds
        
        Returns:
            torch.Tensor: Silent audio (1D)
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)
        silence = torch.zeros(num_samples)
        # Ensure 1D
        silence = self._ensure_1d(silence)
        return silence
    
    def generate_batch_sentences(self, sentences, exaggeration=0.5, cfg_weight=0.5, temperature=0.7):
        """Generate multiple sentences with optimized batching"""
        audio_segments = []
        
        # Process in smaller batches to balance memory and speed
        batch_size = 3 if self.aggressive_memory else 5
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            for sentence in batch:
                if sentence.strip():
                    wav = self.generate_sentence(sentence, exaggeration, cfg_weight, temperature)
                    audio_segments.append(wav)
            
            # Memory cleanup after each batch
            if len(audio_segments) % 10 == 0:  # Every 10 sentences
                import gc
                gc.collect()
        
        return audio_segments
    
    def clean_text_for_basic_tts(self, text):
        """Remove all content within [] tags for basic TTS"""
        cleaned_text = re.sub(r'\[.*?\]', '', text)
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    
    def parse_emotion_tags(self, text):
        """Parse text with emotion tags like [happy] or [sad]"""
        sentences = self.split_into_sentences(text)
        parsed_sentences = []
        
        for sentence in sentences:
            # Look for emotion tags at the beginning of sentences
            emotion_match = re.match(r'^\[(\w+)\]\s*(.*)', sentence)
            if emotion_match:
                emotion = emotion_match.group(1).lower()
                content = emotion_match.group(2)
                
                if emotion in EMOTION_MAPPING:
                    emotion_params = EMOTION_MAPPING[emotion].copy()
                    parsed_sentences.append((content, emotion_params))
                else:
                    print(f"âš ï¸  Unknown emotion '{emotion}', using defaults")
                    parsed_sentences.append((content, {}))
            else:
                parsed_sentences.append((sentence, {}))
        
        return parsed_sentences
    
    def parse_multi_speaker_tags(self, text):
        """Parse multi-speaker dialog with [speaker,emotion] format"""
        parsed_lines = []
        
        # Use regex to find all [speaker] or [speaker,emotion] tags and their content
        pattern = r'\[([^,\]]+)(?:,(\w+))?\](.*?)(?=\[|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for speaker, emotion, content in matches:
            speaker = speaker.strip()
            content = content.strip()
            
            if not content:
                continue
            
            emotion_params = {}
            if emotion and emotion.lower() in EMOTION_MAPPING:
                emotion_params = EMOTION_MAPPING[emotion.lower()].copy()
            elif emotion:
                print(f"âš ï¸  Unknown emotion '{emotion}' for speaker '{speaker}', using defaults")
            
            parsed_lines.append((speaker, content, emotion_params))
        
        # If no matches found, try line-by-line parsing for backward compatibility
        if not parsed_lines:
            lines = text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Handle old format: SPEAKER A: text
                speaker_match = re.match(r'^([^:]+):\s*(.*)', line)
                if speaker_match:
                    speaker = speaker_match.group(1).strip()
                    content = speaker_match.group(2)
                    parsed_lines.append((speaker, content, {}))
                else:
                    # No speaker tag, treat as narrator
                    parsed_lines.append(("narrator", line, {}))
        
        return parsed_lines
    
    def narrate_story(
        self,
        story_text,
        output_path="story_narration.wav",
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.7,
        pause_between_sentences_ms=500,
        emotion_tags=None,
        progress_callback=None
    ):
        """
        Narrate an entire story with sentence-by-sentence processing
        
        Args:
            story_text: The full story text
            output_path: Where to save the audio
            exaggeration: Default emotional intensity (0.25-2.0)
            cfg_weight: Default pacing control (0.2-1.0)
            temperature: Default expressiveness (0.0-1.0)
            pause_between_sentences_ms: Pause duration between sentences
            emotion_tags: Dict mapping sentence indices to emotion settings
                         e.g., {0: {"exaggeration": 1.5}, 5: {"exaggeration": 0.3}}
        
        Returns:
            Path to generated audio file
        """
        print(f"\nðŸ“– Starting story narration...")
        print(f"   Output: {output_path}")
        print(f"   Default emotion: exaggeration={exaggeration}, cfg_weight={cfg_weight}")
        
        # Split into sentences
        sentences = self.split_into_sentences(story_text)
        total_sentences = len(sentences)
        print(f"   ðŸ“ Found {total_sentences} sentences")
        
        if total_sentences > 100:
            estimated_minutes = (total_sentences * 2.0) / 60  # Target 2s per sentence
            print(f"   ðŸŽ¯ Target completion time: ~{estimated_minutes:.0f} minutes")
            print(f"   â³ First 10 sentences will be slower (model warmup)")
        print()
        
        # Process each sentence with CPU optimizations
        audio_segments = []
        start_time = time.time()
        
        # CPU-specific optimizations
        import gc
        torch.set_grad_enabled(False)  # Disable gradients for inference
        
        # Enable aggressive optimizations for long texts
        if total_sentences > 50:
            print("   ðŸš€ Long text detected - enabling aggressive CPU optimizations")
            torch.backends.mkldnn.enabled = True  # Enable Intel MKL-DNN
            if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
                torch.backends.mkl.enabled = True
            
            # Additional CPU optimizations
            torch.set_flush_denormal(True)  # Improve floating point performance
            
            print(f"   âš¡ CPU optimizations active (targeting 2s/sentence)")
        
        # Track timing for ETA estimation
        sentence_times = []
        
        for idx, sentence in enumerate(sentences):
            sentence_start_time = time.time()
            
            # Get emotion settings for this sentence
            sent_exag = exaggeration
            sent_cfg = cfg_weight
            sent_temp = temperature
            
            if emotion_tags and idx in emotion_tags:
                sent_exag = emotion_tags[idx].get("exaggeration", exaggeration)
                sent_cfg = emotion_tags[idx].get("cfg_weight", cfg_weight)
                sent_temp = emotion_tags[idx].get("temperature", temperature)
                print(f"  ðŸŽ­ Sentence {idx+1}/{total_sentences} [EMOTION OVERRIDE]")
            else:
                print(f"  ðŸ“ Sentence {idx+1}/{total_sentences}")
            
            # Performance optimization after warmup
            if idx == 10 and total_sentences > 50:
                print(f"     ðŸš€ Processing optimized after warmup")
            
            # Performance hint every 25 sentences
            if (idx + 1) % 25 == 0 and len(sentence_times) > 10:
                avg_recent = sum(sentence_times[-10:]) / min(len(sentence_times), 10)
                print(f"     âš¡ Recent avg: {avg_recent:.1f}s/sentence")
            
            # Progress and ETA calculation
            if idx > 0:
                progress_percent = ((idx + 1) / total_sentences) * 100
                
                # ETA calculation with better warmup handling
                if idx >= 3:  # Start ETA after 3 sentences for some accuracy
                    if idx < 15:  # During warmup phase - use conservative estimate
                        # Use target performance instead of current slow performance
                        target_time_per_sentence = 3.0  # Target 3 seconds per sentence after warmup
                        remaining_sentences = total_sentences - (idx + 1)
                        eta_seconds = target_time_per_sentence * remaining_sentences
                        eta_str = f"{eta_seconds/60:.1f}m (warming up)"
                    else:  # After warmup - use actual recent performance
                        # Use recent performance instead of all-time average
                        recent_times = sentence_times[-5:] if len(sentence_times) >= 5 else sentence_times[-3:]
                        avg_time_per_sentence = sum(recent_times) / len(recent_times)
                        
                        # Apply slight performance improvement factor
                        speed_factor = 0.9  # Expect 10% speed improvement
                        adjusted_time_per_sentence = avg_time_per_sentence * speed_factor
                        remaining_sentences = total_sentences - (idx + 1)
                        eta_seconds = adjusted_time_per_sentence * remaining_sentences
                        eta_minutes = eta_seconds / 60
                        
                        if eta_minutes > 1:
                            eta_str = f"{eta_minutes:.1f}m"
                        else:
                            eta_str = f"{eta_seconds:.0f}s"
                else:
                    # Very early sentences - show realistic estimate
                    remaining_sentences = total_sentences - (idx + 1)
                    optimistic_time = remaining_sentences * 3.0  # 3 seconds per sentence target
                    eta_str = f"{optimistic_time/60:.1f}m (starting up)"
                
                print(f"     ðŸ“Š Progress: {progress_percent:.1f}% | ETA: {eta_str}")
                
                # Call progress callback if provided (for GUI)
                if progress_callback:
                    progress_callback(progress_percent, eta_str, f"Processing sentence {idx+1}/{total_sentences}")
            
            # Generate audio for this sentence
            wav = self.generate_sentence(
                sentence,
                exaggeration=sent_exag,
                cfg_weight=sent_cfg,
                temperature=sent_temp
            )
            
            # Verify it's 1D before adding
            if wav.dim() != 1:
                print(f"     âš ï¸  Warning: wav is {wav.dim()}D, forcing to 1D")
                wav = self._ensure_1d(wav)
            
            audio_segments.append(wav)
            
            # Add pause (except after last sentence)
            if idx < total_sentences - 1:
                pause = self.add_pause(pause_between_sentences_ms)
                audio_segments.append(pause)
            
            # Track timing
            sentence_time = time.time() - sentence_start_time
            sentence_times.append(sentence_time)
            
            # Warmup completion detection
            if idx == 10:
                recent_avg = sum(sentence_times[-5:]) / 5
                print(f"     ðŸ”¥ Warmup complete! Recent speed: {recent_avg:.1f}s/sentence")
            
            # CPU memory management - adjust frequency based on system
            cleanup_frequency = 5 if self.aggressive_memory else 10
            if (idx + 1) % cleanup_frequency == 0:
                gc.collect()
                print(f"     ðŸ§¹ Memory cleanup (sentence {idx+1})")
        
        # Final progress update
        if progress_callback:
            progress_callback(100, "0s", "Finalizing audio...")
        
        # Concatenate all audio segments
        print(f"\nðŸ”— Concatenating {len(audio_segments)} audio segments...")
        
        # Final safety check - ensure ALL segments are 1D
        for i, seg in enumerate(audio_segments):
            if seg.dim() != 1:
                print(f"   âš ï¸  Segment {i} is {seg.dim()}D (shape: {seg.shape}), converting to 1D...")
                audio_segments[i] = self._ensure_1d(seg)
        
        try:
            full_audio = torch.cat(audio_segments)
        except RuntimeError as e:
            print(f"\nâŒ Error concatenating audio segments: {e}")
            print("   Segment dimensions:")
            for i, seg in enumerate(audio_segments):
                print(f"     Segment {i}: {seg.shape}")
            raise
        
        # Save the final audio
        print(f"ðŸ’¾ Saving to {output_path}...")
        # Ensure we have the right shape for saving (channels, samples)
        if full_audio.dim() == 1:
            full_audio = full_audio.unsqueeze(0)  # Add channel dimension
        ta.save(output_path, full_audio, self.sample_rate)
        
        duration = full_audio.shape[-1] / self.sample_rate
        elapsed = time.time() - start_time
        
        print(f"\nâœ… Story narration complete!")
        print(f"   Duration: {duration/60:.1f} minutes ({duration:.1f} seconds)")
        print(f"   Processing time: {elapsed/60:.1f} minutes")
        print(f"   Real-time factor: {duration/elapsed:.2f}x")
        
        return output_path


def main():
    """Example usage"""
    
    # Example story with varied emotions
    story = """
    It was a dark and stormy night. Thunder crashed across the sky as rain pounded the windows.
    Sarah stood at the edge of the cliff, her heart racing. She had never been so terrified in her life.
    But then, through the storm, she saw a light. A warm, gentle glow in the distance.
    "Could it be?" she whispered, hope rising in her chest.
    She took a deep breath and smiled. Everything was going to be okay.
    """
    
    # Initialize narrator
    narrator = StoryNarrator(
        voice_sample_path=None,  # Optional: use your reference voice
        device="cuda"  # or "cpu"
    )
    
    # Define emotion changes for specific sentences
    emotion_tags = {
        0: {"exaggeration": 0.8, "cfg_weight": 0.4},  # Dramatic opening
        1: {"exaggeration": 1.5, "cfg_weight": 0.3},  # Intense fear
        2: {"exaggeration": 1.2, "cfg_weight": 0.4},  # Terror
        3: {"exaggeration": 0.6, "cfg_weight": 0.6},  # Soft hope
        4: {"exaggeration": 0.4, "cfg_weight": 0.7},  # Calm relief
    }
    
    # Narrate the story
    narrator.narrate_story(
        story_text=story,
        output_path="my_story_narration.wav",
        exaggeration=0.5,           # Default emotion level
        cfg_weight=0.5,             # Default pacing
        temperature=0.7,            # Default expressiveness
        pause_between_sentences_ms=500,  # Half-second pauses
        emotion_tags=emotion_tags   # Per-sentence emotion control
    )


if __name__ == "__main__":
    main()
