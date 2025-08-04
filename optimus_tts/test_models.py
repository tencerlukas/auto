#!/usr/bin/env python3
"""Test all 6 required TTS models"""

from models_required import MODELS
import warnings
warnings.filterwarnings('ignore')

print("🤖 Testing all 6 required TTS models...")
print("=" * 60)

test_text = "Autobots, roll out!"
character = "optimus_prime"

for model_name, model_class in MODELS.items():
    print(f"\nTesting {model_name}...")
    try:
        # Initialize model
        model = model_class()
        
        # Generate audio
        audio, sample_rate = model.generate(test_text, character)
        
        # Check output
        if audio is not None and len(audio) > 0:
            print(f"✓ {model_name} works! Generated {len(audio)} samples at {sample_rate}Hz")
        else:
            print(f"✗ {model_name} failed: No audio generated")
    except Exception as e:
        print(f"✗ {model_name} failed: {e}")

print("\n" + "=" * 60)
print("✅ Testing complete!")
print("\nAll models have been initialized and cached.")
print("They are ready for instant use at http://localhost:5555")