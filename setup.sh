#!/bin/bash
# Setup script for Whiteboard Processing Pipeline

set -e  # Exit on any error

echo "🚀 Setting up Whiteboard Processing Pipeline"
echo "============================================="

# Check Python version
echo "📍 Checking Python version..."
python3 --version

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists, skipping creation"
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   Detected macOS - using macOS-compatible requirements"
    pip install -r requirements_macos.txt
else
    echo "   Detected Linux - using full requirements"
    pip install -r requirements.txt
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "import paddleocr; print('✅ PaddleOCR installed')"
python -c "import easyocr; print('✅ EasyOCR installed')"
python -c "import pytest; print('✅ pytest installed')"
python -c "import structlog; print('✅ structlog installed')"

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Set environment variables:"
echo "      export OPENAI_API_KEY='your-openai-api-key-here'"
echo ""
echo "   2. Activate the environment:"
echo "      source venv/bin/activate"
echo ""
echo "   3. Run examples:"
echo "      python simple_examples.py"
echo ""
echo "   4. Run tests:"
echo "      pytest test_simple_pipeline.py -v"
echo ""
echo "🎯 Focus: Sketch → Mermaid pipeline ready for development!"