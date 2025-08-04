#!/bin/bash

echo "🤖 Setting up Optimus Prime TTS Environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core requirements
echo "Installing core dependencies..."
pip install flask flask-cors numpy scipy soundfile librosa

# Install PyTorch (CPU version for simplicity)
echo "Installing PyTorch..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TTS libraries
echo "Installing TTS libraries..."
pip install gtts pyttsx3 edge-tts

# Install audio processing
echo "Installing audio processing libraries..."
pip install pydub simpleaudio pedalboard

# Try to install optional dependencies
echo "Installing optional dependencies..."
pip install TTS --no-dependencies 2>/dev/null || echo "Coqui TTS not available"
pip install speechbrain --no-dependencies 2>/dev/null || echo "SpeechBrain not available"

# Install espeak if available
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y espeak
elif command -v brew &> /dev/null; then
    brew install espeak
fi

echo "✅ Setup complete!"
echo ""
echo "To run the server:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open http://localhost:5000 in your browser"