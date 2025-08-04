"""TTS Models manager for Transformers voice generation"""

import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import Optional, Tuple, Dict, Any
import tempfile
from abc import ABC, abstractmethod

# Audio processing imports
from scipy import signal
from scipy.io import wavfile
import librosa
from pedalboard import Pedalboard, Reverb, Distortion, PitchShift, Chorus, Delay, Compressor, Gain, Phaser, HighpassFilter, LowpassFilter
from pedalboard.io import AudioFile


# Character voice profiles
CHARACTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "optimus_prime": {
        "name": "Optimus Prime",
        "pitch_shift": -3,
        "distortion": 15,
        "reverb_size": 0.75,
        "reverb_wet": 0.3,
        "chorus_depth": 0.3,
        "delay_time": 0.05,
        "formant_shift": 0.85,
        "modulation_freq": 5,
        "modulation_depth": 0.2,
        "compression_ratio": 4,
        "gain_db": 3,
        "description": "Noble Autobot leader - deep, commanding voice"
    },
    "megatron": {
        "name": "Megatron",
        "pitch_shift": -5,
        "distortion": 25,
        "reverb_size": 0.85,
        "reverb_wet": 0.4,
        "chorus_depth": 0.2,
        "delay_time": 0.08,
        "formant_shift": 0.75,
        "modulation_freq": 3,
        "modulation_depth": 0.3,
        "compression_ratio": 6,
        "gain_db": 5,
        "description": "Decepticon leader - menacing, powerful voice"
    },
    "bumblebee": {
        "name": "Bumblebee",
        "pitch_shift": 3,
        "distortion": 8,
        "reverb_size": 0.5,
        "reverb_wet": 0.2,
        "chorus_depth": 0.5,
        "delay_time": 0.02,
        "formant_shift": 1.2,
        "modulation_freq": 10,
        "modulation_depth": 0.4,
        "compression_ratio": 3,
        "gain_db": 2,
        "radio_effect": True,
        "description": "Young Autobot scout - playful, radio-like voice"
    },
    "starscream": {
        "name": "Starscream",
        "pitch_shift": 2,
        "distortion": 18,
        "reverb_size": 0.6,
        "reverb_wet": 0.25,
        "chorus_depth": 0.4,
        "delay_time": 0.03,
        "formant_shift": 1.1,
        "modulation_freq": 7,
        "modulation_depth": 0.25,
        "compression_ratio": 5,
        "gain_db": 4,
        "description": "Decepticon air commander - shrill, treacherous voice"
    },
    "soundwave": {
        "name": "Soundwave",
        "pitch_shift": -2,
        "distortion": 12,
        "reverb_size": 0.9,
        "reverb_wet": 0.5,
        "chorus_depth": 0.6,
        "delay_time": 0.1,
        "formant_shift": 0.7,
        "modulation_freq": 2,
        "modulation_depth": 0.5,
        "compression_ratio": 8,
        "gain_db": 2,
        "vocoder_effect": True,
        "description": "Decepticon communications - monotone, vocoded voice"
    },
    "jazz": {
        "name": "Jazz",
        "pitch_shift": -1,
        "distortion": 10,
        "reverb_size": 0.65,
        "reverb_wet": 0.25,
        "chorus_depth": 0.35,
        "delay_time": 0.04,
        "formant_shift": 0.95,
        "modulation_freq": 6,
        "modulation_depth": 0.15,
        "compression_ratio": 3,
        "gain_db": 2,
        "description": "Cool Autobot - smooth, laid-back voice"
    },
    "shockwave": {
        "name": "Shockwave",
        "pitch_shift": -4,
        "distortion": 20,
        "reverb_size": 0.8,
        "reverb_wet": 0.35,
        "chorus_depth": 0.1,
        "delay_time": 0.06,
        "formant_shift": 0.8,
        "modulation_freq": 1,
        "modulation_depth": 0.1,
        "compression_ratio": 7,
        "gain_db": 4,
        "description": "Decepticon scientist - cold, logical voice"
    },
    "ratchet": {
        "name": "Ratchet",
        "pitch_shift": -2,
        "distortion": 8,
        "reverb_size": 0.6,
        "reverb_wet": 0.2,
        "chorus_depth": 0.2,
        "delay_time": 0.03,
        "formant_shift": 0.9,
        "modulation_freq": 4,
        "modulation_depth": 0.15,
        "compression_ratio": 3,
        "gain_db": 2,
        "description": "Autobot medic - gruff but caring voice"
    },
    "ironhide": {
        "name": "Ironhide",
        "pitch_shift": -4,
        "distortion": 12,
        "reverb_size": 0.7,
        "reverb_wet": 0.25,
        "chorus_depth": 0.25,
        "delay_time": 0.04,
        "formant_shift": 0.82,
        "modulation_freq": 3,
        "modulation_depth": 0.2,
        "compression_ratio": 5,
        "gain_db": 3,
        "description": "Autobot weapons specialist - tough, grizzled voice"
    },
    "arcee": {
        "name": "Arcee",
        "pitch_shift": 5,
        "distortion": 6,
        "reverb_size": 0.55,
        "reverb_wet": 0.22,
        "chorus_depth": 0.4,
        "delay_time": 0.025,
        "formant_shift": 1.3,
        "modulation_freq": 8,
        "modulation_depth": 0.2,
        "compression_ratio": 3,
        "gain_db": 2,
        "description": "Female Autobot warrior - agile, determined voice"
    }
}


class TTSModel(ABC):
    """Base class for TTS models"""
    
    @abstractmethod
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        """Generate audio from text"""
        pass
    
    def apply_transformer_effect(self, audio: np.ndarray, sample_rate: int, character: str = "optimus_prime") -> np.ndarray:
        """Apply Transformer character voice effects"""
        # Get character profile
        profile = CHARACTER_PROFILES.get(character, CHARACTER_PROFILES["optimus_prime"])
        return self._apply_voice_profile(audio, sample_rate, profile)
    
    def apply_optimus_effect(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply Optimus Prime voice effects (legacy method)"""
        return self.apply_transformer_effect(audio, sample_rate, "optimus_prime")
    
    def _apply_voice_profile(self, audio: np.ndarray, sample_rate: int, profile: dict) -> np.ndarray:
        """Apply specific voice profile to audio"""
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Build effects chain based on character profile
        effects = []
        
        # Pitch shifting
        effects.append(PitchShift(semitones=profile["pitch_shift"]))
        
        # Distortion for metallic sound
        effects.append(Distortion(drive_db=profile["distortion"]))
        
        # Compression
        effects.append(Compressor(
            threshold_db=-20, 
            ratio=profile["compression_ratio"]
        ))
        
        # Reverb for space
        effects.append(Reverb(
            room_size=profile["reverb_size"], 
            damping=0.5, 
            wet_level=profile["reverb_wet"]
        ))
        
        # Chorus for texture
        effects.append(Chorus(
            rate_hz=0.5, 
            depth=profile["chorus_depth"], 
            mix=0.5
        ))
        
        # Delay for echo
        effects.append(Delay(
            delay_seconds=profile["delay_time"], 
            mix=0.2
        ))
        
        # Special effects for specific characters
        if profile.get("radio_effect"):  # Bumblebee
            effects.append(HighpassFilter(cutoff_frequency_hz=300))
            effects.append(LowpassFilter(cutoff_frequency_hz=3000))
            effects.append(Distortion(drive_db=5))
        
        if profile.get("vocoder_effect"):  # Soundwave
            effects.append(Phaser(rate_hz=0.5, depth=0.9, mix=0.7))
            effects.append(Chorus(rate_hz=0.2, depth=0.8, mix=0.6))
        
        # Final gain
        effects.append(Gain(gain_db=profile["gain_db"]))
        
        # Create and apply pedalboard
        board = Pedalboard(effects)
        effected = board(audio, sample_rate)
        
        # Add character-specific modulation
        t = np.linspace(0, len(effected) / sample_rate, len(effected))
        mod_depth = profile["modulation_depth"]
        mod_freq = profile["modulation_freq"]
        
        if profile.get("vocoder_effect"):  # Soundwave gets stepped modulation
            # Create stepped/quantized modulation for vocoder effect
            modulation = 0.7 + mod_depth * np.sign(np.sin(2 * np.pi * mod_freq * t))
        else:
            # Smooth sine modulation for others
            modulation = (1 - mod_depth/2) + mod_depth/2 * np.sin(2 * np.pi * mod_freq * t)
        
        effected = effected * modulation
        
        # Add formant shifting for robotic quality
        effected = self._formant_shift(effected, sample_rate, shift_factor=profile["formant_shift"])
        
        # Additional processing for specific characters
        if profile.get("radio_effect"):  # Bumblebee - add static/noise
            noise = np.random.normal(0, 0.002, len(effected))
            effected = effected + noise
        
        return effected
    
    def _formant_shift(self, audio: np.ndarray, sr: int, shift_factor: float = 0.9) -> np.ndarray:
        """Shift formants to create robotic voice"""
        # Use phase vocoder for formant shifting
        stft = librosa.stft(audio)
        stft_shifted = librosa.phase_vocoder(stft, rate=shift_factor)
        audio_shifted = librosa.istft(stft_shifted)
        
        # Ensure same length
        if len(audio_shifted) > len(audio):
            audio_shifted = audio_shifted[:len(audio)]
        elif len(audio_shifted) < len(audio):
            audio_shifted = np.pad(audio_shifted, (0, len(audio) - len(audio_shifted)))
        
        return audio_shifted


class CoquiTTS(TTSModel):
    """Coqui TTS implementation - using pyttsx3 as fallback"""
    
    def __init__(self):
        self.tts = None
        self.use_fallback = True
        try:
            # Try to import TTS
            from TTS.api import TTS
            # List available models
            models = TTS.list_models()
            if models:
                # Try to use a simple English model
                for model in ["tts_models/en/ljspeech/tacotron2-DDC", 
                             "tts_models/en/ljspeech/glow-tts",
                             "tts_models/en/ljspeech/speedy-speech"]:
                    if model in models:
                        self.tts = TTS(model)
                        self.use_fallback = False
                        print(f"Loaded Coqui TTS model: {model}")
                        break
                if self.use_fallback:
                    # Try any English model
                    en_models = [m for m in models if 'en' in m.lower()]
                    if en_models:
                        self.tts = TTS(en_models[0])
                        self.use_fallback = False
                        print(f"Loaded Coqui TTS model: {en_models[0]}")
        except Exception as e:
            print(f"Coqui TTS not available, using pyttsx3 fallback: {e}")
        
        if self.use_fallback:
            # Use pyttsx3 as fallback
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 1.0)
                # Try to use a male voice
                voices = self.engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
            except:
                self.engine = None
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if not self.use_fallback and self.tts is not None:
            # Use actual Coqui TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                try:
                    self.tts.tts_to_file(text=text, file_path=tmp_file.name)
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                    return self.apply_transformer_effect(audio, sr, character), sr
                except Exception as e:
                    print(f"Coqui TTS generation failed: {e}")
                    return self._simple_fallback(text, character)
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
        elif self.engine is not None:
            # Use pyttsx3 fallback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                try:
                    self.engine.save_to_file(text, tmp_file.name)
                    self.engine.runAndWait()
                    if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                        audio, sr = librosa.load(tmp_file.name, sr=None)
                        return self.apply_transformer_effect(audio, sr, character), sr
                    else:
                        return self._simple_fallback(text, character)
                except Exception as e:
                    print(f"Pyttsx3 fallback failed: {e}")
                    return self._simple_fallback(text, character)
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
        else:
            return self._simple_fallback(text, character)
    
    def _simple_fallback(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        """Simple synthesized voice fallback"""
        # Generate a more complex waveform for better quality
        duration = min(len(text) * 0.08, 5.0)
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        
        # Mix multiple frequencies for richer sound
        fundamental = 150  # Base frequency
        audio = np.zeros_like(t)
        
        # Add harmonics
        for harmonic in [1, 2, 3, 4, 5]:
            amplitude = 1.0 / harmonic
            audio += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
        
        # Add some envelope
        envelope = np.exp(-t * 0.5) * 0.5
        audio = audio * envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.5
        
        return self.apply_transformer_effect(audio, sr, character), sr


class GTTS(TTSModel):
    """Google TTS implementation"""
    
    def __init__(self):
        from gtts import gTTS
        self.gTTS = gTTS
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            try:
                tts = self.gTTS(text=text, lang='en', slow=False)
                tts.save(tmp_file.name)
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                os.unlink(tmp_file.name)


class Pyttsx3TTS(TTSModel):
    """Pyttsx3 offline TTS implementation"""
    
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        # Set properties for more robotic voice
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to use a male voice
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                self.engine.save_to_file(text, tmp_file.name)
                self.engine.runAndWait()
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class EdgeTTS(TTSModel):
    """Microsoft Edge TTS implementation"""
    
    def __init__(self):
        self.voice = "en-US-ChristopherNeural"  # Deep male voice
    
    async def generate_async(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import edge_tts
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(tmp_file.name)
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                os.unlink(tmp_file.name)
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_async(text, character))
        finally:
            loop.close()


class SileroTTS(TTSModel):
    """Silero TTS implementation"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        try:
            self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                          model='silero_tts',
                                          language='en',
                                          speaker='v3_en')
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Could not load Silero: {e}")
            self.model = None
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if self.model is None:
            # Fallback
            duration = min(len(text) * 0.1, 5.0)
            sr = 48000
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * 200 * t) * 0.5
            return self.apply_transformer_effect(audio, sr, character), sr
        
        audio = self.model.apply_tts(text=text, speaker='en_0', sample_rate=48000)
        audio_np = audio.numpy()
        return self.apply_transformer_effect(audio_np, 48000, character), 48000


class BarkTTS(TTSModel):
    """Bark TTS implementation"""
    
    def __init__(self):
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            preload_models()
            self.generate_audio = generate_audio
            self.sample_rate = SAMPLE_RATE
        except Exception as e:
            print(f"Could not load Bark: {e}")
            self.generate_audio = None
            self.sample_rate = 24000
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if self.generate_audio is None:
            # Fallback
            duration = min(len(text) * 0.1, 5.0)
            sr = self.sample_rate
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * 200 * t) * 0.5
            return self.apply_transformer_effect(audio, sr, character), sr
        
        # Add voice preset for deeper voice
        text_prompt = f"[MAN] {text}"
        audio = self.generate_audio(text_prompt)
        return self.apply_transformer_effect(audio, self.sample_rate, character), self.sample_rate


class EspeakTTS(TTSModel):
    """eSpeak TTS implementation - lightweight and robotic by nature"""
    
    def __init__(self):
        self.speed = 150
        self.pitch = 30  # Lower pitch for deeper voice
        self.voice = "en+m3"  # Male voice variant
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                cmd = [
                    'espeak',
                    '-v', self.voice,
                    '-s', str(self.speed),
                    '-p', str(self.pitch),
                    '-w', tmp_file.name,
                    text
                ]
                subprocess.run(cmd, check=False, capture_output=True)
                
                if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                    return self.apply_transformer_effect(audio, sr, character), sr
                else:
                    # Fallback if espeak fails
                    return self._generate_fallback(text, character)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def _generate_fallback(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        duration = min(len(text) * 0.1, 5.0)
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 200 * t) * 0.5
        return self.apply_transformer_effect(audio, sr, character), sr


# Model registry
MODELS = {
    "Neural TTS (System)": CoquiTTS,  # Falls back to pyttsx3 if Coqui not available
    "Google TTS": GTTS,
    "Pyttsx3 (System)": Pyttsx3TTS,
    "Edge TTS (Christopher)": EdgeTTS,
    "Silero TTS": SileroTTS,
    "Bark TTS": BarkTTS,
    "eSpeak (Robotic)": EspeakTTS,
}

# Export character profiles for API use
__all__ = ['MODELS', 'CHARACTER_PROFILES']