"""
Simple Chatterbox Story Example - Ready to Run!
"""

import torch

# Check if chatterbox is installed
try:
    from chatterbox_story_tts import StoryNarrator
    print("✅ Chatterbox story narrator imported successfully!")
except ImportError:
    print("❌ Please install chatterbox first:")
    print("   pip install chatterbox-tts --break-system-packages")
    exit(1)

# Check GPU
print(f"\n🖥️  System Check:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Short test story with emotional variety
test_story = """
The old lighthouse stood alone on the rocky cliff, its light cutting through the fog.
Captain Morgan gripped the railing, his knuckles white with tension.
"We're not going to make it!" shouted the first mate over the howling wind.
But Morgan smiled, a calm certainty washing over him.
"Yes we will," he said softly. "I've sailed through worse than this."
The crew fell silent, their fear melting into hope.
As dawn broke, the storm passed, leaving the sea glass-smooth and peaceful.
They had survived another night.
"""

print("\n" + "="*60)
print("🎬 CHATTERBOX STORY NARRATOR - TEST RUN")
print("="*60)

# Initialize narrator
print("\n1️⃣  Initializing narrator...")
narrator = StoryNarrator(
    voice_sample_path=None,  # Using default voice (change to your voice file)
    device="cpu"  # CPU-optimized mode
)

# Define emotional moments in the story
emotion_map = {
    0: {"exaggeration": 0.6, "cfg_weight": 0.25},  # Descriptive opening - VERY SLOW
    1: {"exaggeration": 1.2, "cfg_weight": 0.2},   # Tense moment - VERY SLOW (compensates for high exaggeration)
    2: {"exaggeration": 1.8, "cfg_weight": 0.15},  # Panic! Very dramatic - SLOWEST (compensates for very high exaggeration)
    3: {"exaggeration": 0.4, "cfg_weight": 0.3},   # Calm confidence - SLOW
    4: {"exaggeration": 0.5, "cfg_weight": 0.3},   # Quiet determination - SLOW
    5: {"exaggeration": 0.4, "cfg_weight": 0.3},   # Hope building - SLOW
    6: {"exaggeration": 0.3, "cfg_weight": 0.3},   # Peaceful resolution - SLOW
    7: {"exaggeration": 0.3, "cfg_weight": 0.3},   # Quiet ending - SLOW
}

print("\n2️⃣  Generating story with emotion control...")
print("   📝 Story length: ~45-50 seconds (with slower pacing)")
print("   🎭 Emotion variations: 8 different emotional tones")

# Generate the narration
output_file = narrator.narrate_story(
    story_text=test_story,
    output_path="test_story_narration.wav",
    exaggeration=0.5,                    # Default emotion
    cfg_weight=0.25,                     # Default pacing - EVEN SLOWER (was 0.35)
    temperature=0.7,                     # Default expressiveness
    pause_between_sentences_ms=1200,     # Much longer pauses (was 800)
    emotion_tags=emotion_map             # Our emotion map
)

print("\n" + "="*60)
print("✅ SUCCESS!")
print("="*60)
print(f"\n🎵 Your narration is ready: {output_file}")
print("\n💡 Next Steps:")
print("   1. Listen to test_story_narration.wav")
print("   2. Adjust emotion_map values if needed")
print("   3. Replace test_story with your full story")
print("   4. (Optional) Add voice_sample_path for voice cloning")
print("\n🎭 Emotion Quick Reference:")
print("   exaggeration: 0.3=calm, 0.5=normal, 1.0=dramatic, 1.8=intense")
print("   cfg_weight: 0.3=slow, 0.5=normal, 0.7=fast")
print("\n" + "="*60)
