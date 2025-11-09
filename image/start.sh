#!/bin/bash

echo ""
echo "🎬 Animation Batch Generator - Quick Start"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✅ Python found"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt --quiet

echo ""
echo "🚀 Starting Animation Batch Generator..."
echo ""
echo "The application will open in your browser at:"
echo "   👉 http://127.0.0.1:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
python3 app_final_timeline.py
    pip3 install -r requirements.txt
fi

echo "✅ Dependencies ready"
echo ""

# Check ComfyUI
echo "🔍 Checking ComfyUI connection..."
if curl -s http://127.0.0.1:8000/system_stats > /dev/null 2>&1; then
    echo "✅ ComfyUI is running"
else
    echo "⚠️  Warning: ComfyUI not detected on port 8000"
    echo "   Start ComfyUI before generating images"
fi

echo ""
echo "🚀 Starting Animation Batch Generator..."
echo "   Opening at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 animation_batch_generator.py
