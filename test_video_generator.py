#!/usr/bin/env python3
"""
Quick Test Script for Video Generator
Tests the system with the Episode 01 files
"""

import sys
import os
from pathlib import Path

# Add current directory to path so we can import video_generator
sys.path.insert(0, str(Path(__file__).parent))

try:
    from video_generator import VideoGenerator
    print("✅ Video generator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import video generator: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements_video.txt")
    exit(1)

def test_file_parsing():
    """Test parsing the Episode 01 files"""
    print("\n🔍 Testing file parsing...")
    
    script_file = "Episode_01_Script_The_Wanderer_EXPANDED.txt"
    prompts_file = "Episode_01_Image_Prompts_The_Wanderer_EXPANDED.txt"
    
    if not os.path.exists(script_file):
        print(f"❌ Script file not found: {script_file}")
        return False
        
    if not os.path.exists(prompts_file):
        print(f"❌ Prompts file not found: {prompts_file}")
        return False
    
    generator = VideoGenerator()
    
    try:
        scenes, characters = generator.parse_episode_files(script_file, prompts_file)
        
        print(f"✅ Found {len(scenes)} scenes")
        print(f"✅ Found {len(characters)} characters")
        
        # Show first few scenes
        for i, scene in enumerate(scenes[:3]):
            print(f"\nScene {scene['number']}: {scene['title']}")
            print(f"  Script preview: {scene['script_text'][:100]}...")
            print(f"  Image prompts: {len(scene['image_prompts'])}")
        
        # Show characters
        print(f"\nCharacters found:")
        for char_name in characters.keys():
            print(f"  - {char_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing files: {e}")
        return False

def test_comfyui_connection():
    """Test connection to ComfyUI"""
    print("\n🔌 Testing ComfyUI connection...")
    
    generator = VideoGenerator()
    
    try:
        import requests
        response = requests.get(f"{generator.comfyui_url}/system_stats", timeout=5)
        if response.status_code == 200:
            print("✅ ComfyUI connection successful")
            
            # Test getting models
            models = generator.get_available_models()
            print(f"✅ Found {len(models)} available models:")
            for model in models[:5]:  # Show first 5
                print(f"  - {model}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
            
            return True
        else:
            print(f"❌ ComfyUI returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to ComfyUI: {e}")
        print("Make sure ComfyUI is running on http://127.0.0.1:8000")
        return False

def test_audio_system():
    """Test if audio system is available"""
    print("\n🎙️ Testing audio system...")
    
    try:
        from audio.chatterbox_story_tts import StoryNarrator
        print("✅ Audio system (Chatterbox TTS) available")
        
        # Try to initialize (don't actually generate audio in test)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Audio will use: {device}")
        
        return True
    except ImportError as e:
        print(f"⚠️ Audio system not available: {e}")
        print("Audio generation will be skipped in video creation")
        return False

def test_ffmpeg():
    """Test if FFmpeg is available"""
    print("\n🎬 Testing FFmpeg...")
    
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, check=True, timeout=10)
        print("✅ FFmpeg is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ FFmpeg not found")
        print("Please install FFmpeg and add it to your PATH for video creation")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Video Generator Tests")
    print("=" * 50)
    
    tests = [
        ("File Parsing", test_file_parsing),
        ("ComfyUI Connection", test_comfyui_connection),
        ("Audio System", test_audio_system),
        ("FFmpeg", test_ffmpeg)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The video generator should work correctly.")
        print("You can now run: python video_generator.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues above before using the video generator.")
        
        if not results.get("ComfyUI Connection", False):
            print("\n🔧 To fix ComfyUI:")
            print("1. Install and start ComfyUI")
            print("2. Load a Pixar/cartoon model")
            print("3. Ensure it runs on http://127.0.0.1:8000")
        
        if not results.get("FFmpeg", False):
            print("\n🔧 To fix FFmpeg:")
            print("1. Download FFmpeg from https://ffmpeg.org/download.html")
            print("2. Extract and add to your system PATH")
            print("3. Restart your terminal")

if __name__ == "__main__":
    main()