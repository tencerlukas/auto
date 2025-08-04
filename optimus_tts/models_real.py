"""Real TTS Models for Transformers voice generation - No fallbacks!"""

import os
import numpy as np
import torch
import tempfile
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Audio processing imports
from pedalboard import Pedalboard, Reverb, Distortion, PitchShift, Chorus, Delay, Compressor, Gain, Phaser, HighpassFilter, LowpassFilter


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
        profile = CHARACTER_PROFILES.get(character, CHARACTER_PROFILES["optimus_prime"])
        return self._apply_voice_profile(audio, sample_rate, profile)
    
    def _apply_voice_profile(self, audio: np.ndarray, sample_rate: int, profile: dict) -> np.ndarray:
        """Apply specific voice profile to audio"""
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Build effects chain
        effects = []
        effects.append(PitchShift(semitones=profile["pitch_shift"]))
        effects.append(Distortion(drive_db=profile["distortion"]))
        effects.append(Compressor(threshold_db=-20, ratio=profile["compression_ratio"]))
        effects.append(Reverb(room_size=profile["reverb_size"], damping=0.5, wet_level=profile["reverb_wet"]))
        effects.append(Chorus(rate_hz=0.5, depth=profile["chorus_depth"], mix=0.5))
        effects.append(Delay(delay_seconds=profile["delay_time"], mix=0.2))
        
        # Special effects
        if profile.get("radio_effect"):
            effects.append(HighpassFilter(cutoff_frequency_hz=300))
            effects.append(LowpassFilter(cutoff_frequency_hz=3000))
            effects.append(Distortion(drive_db=5))
        
        if profile.get("vocoder_effect"):
            effects.append(Phaser(rate_hz=0.5, depth=0.9, mix=0.7))
            effects.append(Chorus(rate_hz=0.2, depth=0.8, mix=0.6))
        
        effects.append(Gain(gain_db=profile["gain_db"]))
        
        # Apply effects
        board = Pedalboard(effects)
        effected = board(audio, sample_rate)
        
        # Add modulation
        t = np.linspace(0, len(effected) / sample_rate, len(effected))
        mod_depth = profile["modulation_depth"]
        mod_freq = profile["modulation_freq"]
        
        if profile.get("vocoder_effect"):
            modulation = 0.7 + mod_depth * np.sign(np.sin(2 * np.pi * mod_freq * t))
        else:
            modulation = (1 - mod_depth/2) + mod_depth/2 * np.sin(2 * np.pi * mod_freq * t)
        
        effected = effected * modulation
        
        # Formant shifting
        effected = self._formant_shift(effected, sample_rate, shift_factor=profile["formant_shift"])
        
        if profile.get("radio_effect"):
            noise = np.random.normal(0, 0.002, len(effected))
            effected = effected + noise
        
        return effected
    
    def _formant_shift(self, audio: np.ndarray, sr: int, shift_factor: float = 0.9) -> np.ndarray:
        """Shift formants to create robotic voice"""
        stft = librosa.stft(audio)
        stft_shifted = librosa.phase_vocoder(stft, rate=shift_factor)
        audio_shifted = librosa.istft(stft_shifted)
        
        if len(audio_shifted) > len(audio):
            audio_shifted = audio_shifted[:len(audio)]
        elif len(audio_shifted) < len(audio):
            audio_shifted = np.pad(audio_shifted, (0, len(audio) - len(audio_shifted)))
        
        return audio_shifted


class SpeechT5TTS(TTSModel):
    """Microsoft SpeechT5 from Hugging Face"""
    
    def __init__(self):
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset
        
        print("Loading SpeechT5 model from Hugging Face...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Load speaker embeddings (male voice)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        # Use a male speaker embedding
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        print("SpeechT5 loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        inputs = self.processor(text=text, return_tensors="pt")
        
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        
        audio = speech.numpy()
        sample_rate = 16000
        return self.apply_transformer_effect(audio, sample_rate, character), sample_rate


class SileroTTS(TTSModel):
    """Silero TTS - Fast and lightweight"""
    
    def __init__(self):
        print("Loading Silero model...")
        self.device = torch.device('cpu')
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                 model='silero_tts',
                                 language='en',
                                 speaker='v3_en',
                                 trust_repo=True)
        self.model = model.to(self.device)
        print("Silero loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        # Use male speaker
        audio = self.model.apply_tts(text=text, 
                                    speaker='en_0',  # Male voice
                                    sample_rate=48000,
                                    put_accent=True,
                                    put_yo=True)
        audio_np = audio.numpy() if torch.is_tensor(audio) else audio
        return self.apply_transformer_effect(audio_np, 48000, character), 48000


class VITS(TTSModel):
    """VITS model from Meta"""
    
    def __init__(self):
        from transformers import VitsModel, AutoTokenizer
        
        print("Loading VITS model from Hugging Face...")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        print("VITS loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model(**inputs).waveform
        
        audio = output.squeeze().cpu().numpy()
        sample_rate = self.model.config.sampling_rate
        
        return self.apply_transformer_effect(audio, sample_rate, character), sample_rate


class GTTS(TTSModel):
    """Google TTS - Still works as a cloud service"""
    
    def __init__(self):
        from gtts import gTTS
        self.gTTS = gTTS
        print("Google TTS ready!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            try:
                tts = self.gTTS(text=text, lang='en', slow=False)
                tts.save(tmp_file.name)
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class EdgeTTS(TTSModel):
    """Microsoft Edge TTS"""
    
    def __init__(self):
        self.voice = "en-US-ChristopherNeural"  # Deep male voice
        print("Edge TTS ready!")
    
    async def generate_async(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import edge_tts
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(tmp_file.name)
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_async(text, character))
        finally:
            loop.close()


class Pyttsx3TTS(TTSModel):
    """System TTS using pyttsx3"""
    
    def __init__(self):
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
        print("Pyttsx3 ready!")
    
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


class EspeakTTS(TTSModel):
    """eSpeak TTS - Classic robotic voice"""
    
    def __init__(self):
        self.speed = 150
        self.pitch = 30  # Lower pitch for deeper voice
        self.voice = "en+m3"  # Male voice variant
        print("eSpeak ready!")
    
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
                subprocess.run(cmd, check=True, capture_output=True)
                
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


# Model registry - All real models!
MODELS = {
    "SpeechT5 (Microsoft)": SpeechT5TTS,
    "Silero (Fast Neural)": SileroTTS,
    "VITS (Meta MMS)": VITS,
    "Google TTS (Cloud)": GTTS,
    "Edge TTS (Microsoft)": EdgeTTS,
    "System Voice (pyttsx3)": Pyttsx3TTS,
    "eSpeak (Robotic)": EspeakTTS,
}

# Export for API use
__all__ = ['MODELS', 'CHARACTER_PROFILES']