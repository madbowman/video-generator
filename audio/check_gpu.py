"""
Check if Chatterbox is using GPU properly
"""

import torch
import sys

print("=" * 60)
print("GPU DIAGNOSTIC FOR CHATTERBOX")
print("=" * 60)

# Check PyTorch CUDA
print("\n1. PyTorch CUDA Status:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Current device: {torch.cuda.current_device()}")
else:
    print("   ❌ CUDA NOT AVAILABLE - Running on CPU!")
    print("\n   Common causes:")
    print("   - PyTorch CPU-only version installed")
    print("   - NVIDIA drivers not installed")
    print("   - CUDA toolkit mismatch")

# Check if Chatterbox can load
print("\n2. Chatterbox Model Check:")
try:
    from chatterbox.tts import ChatterboxTTS
    print("   ✅ Chatterbox installed")
    
    # Try to load model
    print("\n3. Loading model on GPU...")
    model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Check where model actually is
    model_device = next(model.parameters()).device
    print(f"   Model loaded on: {model_device}")
    
    if torch.cuda.is_available():
        if "cuda" in str(model_device):
            print("   ✅ Model is ON GPU!")
        else:
            print("   ❌ Model is ON CPU despite CUDA being available!")
    
    # Check VRAM usage
    if torch.cuda.is_available():
        print(f"\n4. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    print("\n5. Quick Generation Test:")
    import time
    start = time.time()
    wav = model.generate("This is a test of GPU acceleration.")
    elapsed = time.time() - start
    
    duration = len(wav) / model.sr
    rtf = duration / elapsed
    
    print(f"   Generated {duration:.1f}s audio in {elapsed:.1f}s")
    print(f"   Real-time factor: {rtf:.2f}x")
    
    if rtf > 2.0:
        print("   ✅ GOOD - Running fast (GPU likely working)")
    elif rtf > 1.0:
        print("   ⚠️  OK - Slightly slow (check GPU usage)")
    else:
        print("   ❌ SLOW - Likely running on CPU!")
    
except ImportError:
    print("   ❌ Chatterbox not installed")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Final recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if not torch.cuda.is_available():
    print("""
❌ CUDA NOT AVAILABLE - Install CUDA-enabled PyTorch:

For CUDA 12.1:
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

For CUDA 11.8:
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Then restart Python and run this script again.
""")
elif torch.cuda.is_available():
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device="cuda")
        model_device = next(model.parameters()).device
        
        if "cuda" in str(model_device):
            print("""
✅ GPU IS WORKING!

If generation is still slow:
1. Close other GPU applications
2. Check Task Manager → Performance → GPU
3. Try larger batch/parallel processing
4. Ensure model stays on GPU between generations
""")
        else:
            print("""
⚠️  CUDA available but model on CPU!

Fix: Explicitly set device in code:
    model = ChatterboxTTS.from_pretrained(device="cuda")
    model = model.to("cuda")
""")
    except:
        pass

print("\n" + "=" * 60)
input("Press Enter to exit...")
