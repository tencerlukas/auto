"""Real TTS Models for Transformers voice generation - No fallbacks!"""

import os
import numpy as np
import torch
import tempfile
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
import librosa
import soundfile as sf

# Audio processing imports
from pedalboard import Pedalboard, Reverb, Distortion, PitchShift, Chorus, Delay, Compressor, Gain, Phaser, HighpassFilter, LowpassFilter


# Character voice profiles (same as before)
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
        
        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        print("SpeechT5 loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        audio = speech.numpy()
        sample_rate = 16000
        return self.apply_transformer_effect(audio, sample_rate, character), sample_rate


class BarkTTS(TTSModel):
    """Bark TTS from Suno AI"""
    
    def __init__(self):
        from transformers import AutoProcessor, BarkModel
        
        print("Loading Bark model from Hugging Face...")
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")
        self.model = self.model.to("cpu")
        print("Bark loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        # Use voice preset for male voice
        voice_preset = "v2/en_speaker_6"  # Male voice
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        audio = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return self.apply_transformer_effect(audio, sample_rate, character), sample_rate


class ParlerTTS(TTSModel):
    """Parler TTS from Hugging Face"""
    
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForTextToWaveform
        
        print("Loading Parler TTS model from Hugging Face...")
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        self.model = AutoModelForTextToWaveform.from_pretrained("parler-tts/parler-tts-mini-v1")
        print("Parler TTS loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        description = "A male speaker with a deep, clear voice speaking at a moderate speed"
        
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio = generation.cpu().numpy().squeeze()
        sample_rate = self.model.config.sampling_rate
        
        return self.apply_transformer_effect(audio, sample_rate, character), sample_rate


class SileroTTS(TTSModel):
    """Silero TTS - Fast and lightweight"""
    
    def __init__(self):
        print("Loading Silero model...")
        self.device = torch.device('cpu')
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language='en',
                                      speaker='v3_en')
        self.model = self.model.to(self.device)
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
    """VITS model from Hugging Face"""
    
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


class WhisperSpeechTTS(TTSModel):
    """Using Transformers pipeline for TTS"""
    
    def __init__(self):
        from transformers import pipeline
        
        print("Loading TTS pipeline...")
        self.synthesizer = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
        print("TTS pipeline loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        speech = self.synthesizer(text)
        audio = speech["audio"].squeeze()
        sample_rate = speech["sampling_rate"]
        
        return self.apply_transformer_effect(audio, sample_rate, character), sample_rate


class FastPitch(TTSModel):
    """Using MMS TTS as alternative"""
    
    def __init__(self):
        from transformers import VitsModel, AutoTokenizer
        
        print("Loading MMS TTS model...")
        # Using a different MMS model
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        print("MMS TTS loaded successfully!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model(**inputs).waveform
        
        audio = output.squeeze().cpu().numpy()
        sample_rate = 16000  # MMS uses 16kHz
        
        return self.apply_transformer_effect(audio, sample_rate, character), sample_rate


# Model registry - All real models, no fallbacks!
MODELS = {
    "SpeechT5 (Microsoft)": SpeechT5TTS,
    "Bark (Suno AI)": BarkTTS,
    "Parler TTS": ParlerTTS,
    "Silero (Fast)": SileroTTS,
    "VITS (Facebook)": VITS,
    "Kakao VITS": WhisperSpeechTTS,
    "MMS TTS (Meta)": FastPitch,
}

# Export for API use
__all__ = ['MODELS', 'CHARACTER_PROFILES']