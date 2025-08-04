"""Flask app for Optimus Prime TTS"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import base64
import io
import numpy as np
import soundfile as sf
from models_required import MODELS, CHARACTER_PROFILES
import traceback
import os

app = Flask(__name__)
CORS(app)

# Cache for loaded models
model_cache = {}

def get_model(model_name):
    """Get or create model instance"""
    if model_name not in model_cache:
        if model_name in MODELS:
            try:
                model_cache[model_name] = MODELS[model_name]()
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                return None
    return model_cache.get(model_name)


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    return jsonify(list(MODELS.keys()))


@app.route('/api/characters', methods=['GET'])
def get_characters():
    """Get list of available Transformer characters"""
    characters = []
    for key, profile in CHARACTER_PROFILES.items():
        characters.append({
            'id': key,
            'name': profile['name'],
            'description': profile['description']
        })
    return jsonify(characters)


@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    """Synthesize speech from text"""
    try:
        data = request.json
        text = data.get('text', '')
        model_name = data.get('model', list(MODELS.keys())[0])
        character = data.get('character', 'optimus_prime')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Validate character
        if character not in CHARACTER_PROFILES:
            character = 'optimus_prime'
        
        # Get model
        model = get_model(model_name)
        if model is None:
            return jsonify({'error': f'Failed to load model: {model_name}'}), 500
        
        # Generate audio with character voice
        try:
            audio, sample_rate = model.generate(text, character)
        except Exception as e:
            print(f"Error generating with {model_name}: {e}")
            # Fallback to simple generation
            duration = min(len(text) * 0.1, 5.0)
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 200 * t) * 0.5
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'audio': audio_base64,
            'sample_rate': sample_rate
        })
    
    except Exception as e:
        print(f"Error in synthesize: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'models_available': len(MODELS),
        'characters_available': len(CHARACTER_PROFILES)
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting Optimus Prime TTS Server...")
    print(f"Available models: {list(MODELS.keys())}")
    app.run(debug=True, host='0.0.0.0', port=5555)