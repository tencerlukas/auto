"""Flask app for Transformers TTS with complete model list and license info"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
import soundfile as sf
from models_complete import MODELS, CHARACTER_PROFILES
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
                model_class, license_info, is_open_source = MODELS[model_name]
                model_cache[model_name] = model_class()
                print(f"Loaded model: {model_name} (License: {license_info}, Open Source: {is_open_source})")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                return None
    return model_cache.get(model_name)


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index_complete.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models with license info"""
    models_info = []
    for name, (model_class, license_info, is_open_source) in MODELS.items():
        models_info.append({
            'name': name,
            'license': license_info,
            'open_source': is_open_source
        })
    return jsonify(models_info)


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
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
        
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
    total_models = len(MODELS)
    open_source_models = sum(1 for _, (_, _, is_os) in MODELS.items() if is_os)
    proprietary_models = total_models - open_source_models
    
    return jsonify({
        'status': 'healthy',
        'total_models': total_models,
        'open_source_models': open_source_models,
        'proprietary_models': proprietary_models,
        'characters_available': len(CHARACTER_PROFILES)
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting Transformers TTS Server (Complete Edition)...")
    print(f"Total models available: {len(MODELS)}")
    
    # Print model summary
    open_source = [name for name, (_, _, is_os) in MODELS.items() if is_os]
    proprietary = [name for name, (_, _, is_os) in MODELS.items() if not is_os]
    
    print(f"\nOpen Source Models ({len(open_source)}):")
    for model in open_source:
        _, license_info, _ = MODELS[model]
        print(f"  • {model} - {license_info}")
    
    print(f"\nProprietary/Freeware Models ({len(proprietary)}):")
    for model in proprietary:
        _, license_info, _ = MODELS[model]
        print(f"  • {model} - {license_info}")
    
    app.run(debug=True, host='0.0.0.0', port=5555)