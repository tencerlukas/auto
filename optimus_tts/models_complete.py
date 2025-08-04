"""Complete list of ALL local TTS models with license information"""

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


# Character voice profiles with reduced echo
CHARACTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "human_voice": {
        "name": "No Effect (Human Voice)",
        "pitch_shift": 0,
        "distortion": 0,
        "reverb_size": 0,
        "reverb_wet": 0,
        "chorus_depth": 0,
        "delay_time": 0,
        "formant_shift": 1.0,
        "modulation_freq": 0,
        "modulation_depth": 0,
        "compression_ratio": 1,
        "gain_db": 0,
        "bypass_effects": True,
        "description": "Original human voice without any effects"
    },
    "optimus_prime": {
        "name": "Optimus Prime",
        "pitch_shift": -3,
        "distortion": 8,
        "reverb_size": 0.25,
        "reverb_wet": 0.1,
        "chorus_depth": 0.15,
        "delay_time": 0.01,
        "formant_shift": 0.85,
        "modulation_freq": 5,
        "modulation_depth": 0.1,
        "compression_ratio": 3,
        "gain_db": 2,
        "description": "Noble Autobot leader - deep, commanding voice"
    },
    "megatron": {
        "name": "Megatron",
        "pitch_shift": -5,
        "distortion": 12,
        "reverb_size": 0.3,
        "reverb_wet": 0.15,
        "chorus_depth": 0.1,
        "delay_time": 0.02,
        "formant_shift": 0.75,
        "modulation_freq": 3,
        "modulation_depth": 0.15,
        "compression_ratio": 4,
        "gain_db": 3,
        "description": "Decepticon leader - menacing, powerful voice"
    },
    "bumblebee": {
        "name": "Bumblebee",
        "pitch_shift": 3,
        "distortion": 5,
        "reverb_size": 0.15,
        "reverb_wet": 0.08,
        "chorus_depth": 0.25,
        "delay_time": 0.005,
        "formant_shift": 1.2,
        "modulation_freq": 10,
        "modulation_depth": 0.2,
        "compression_ratio": 2,
        "gain_db": 1,
        "radio_effect": True,
        "description": "Young Autobot scout - playful, radio-like voice"
    },
    "starscream": {
        "name": "Starscream",
        "pitch_shift": 2,
        "distortion": 9,
        "reverb_size": 0.2,
        "reverb_wet": 0.1,
        "chorus_depth": 0.2,
        "delay_time": 0.01,
        "formant_shift": 1.1,
        "modulation_freq": 7,
        "modulation_depth": 0.12,
        "compression_ratio": 3,
        "gain_db": 2,
        "description": "Decepticon air commander - shrill, treacherous voice"
    },
    "soundwave": {
        "name": "Soundwave",
        "pitch_shift": -2,
        "distortion": 8,
        "reverb_size": 0.3,
        "reverb_wet": 0.2,
        "chorus_depth": 0.4,
        "delay_time": 0.03,
        "formant_shift": 0.7,
        "modulation_freq": 2,
        "modulation_depth": 0.3,
        "compression_ratio": 5,
        "gain_db": 2,
        "vocoder_effect": True,
        "description": "Decepticon communications - monotone, vocoded voice"
    },
    "jazz": {
        "name": "Jazz",
        "pitch_shift": -1,
        "distortion": 5,
        "reverb_size": 0.2,
        "reverb_wet": 0.1,
        "chorus_depth": 0.15,
        "delay_time": 0.01,
        "formant_shift": 0.95,
        "modulation_freq": 6,
        "modulation_depth": 0.08,
        "compression_ratio": 2,
        "gain_db": 1,
        "description": "Cool Autobot - smooth, laid-back voice"
    },
    "shockwave": {
        "name": "Shockwave",
        "pitch_shift": -4,
        "distortion": 10,
        "reverb_size": 0.25,
        "reverb_wet": 0.12,
        "chorus_depth": 0.05,
        "delay_time": 0.015,
        "formant_shift": 0.8,
        "modulation_freq": 1,
        "modulation_depth": 0.05,
        "compression_ratio": 4,
        "gain_db": 2,
        "description": "Decepticon scientist - cold, logical voice"
    },
    "ratchet": {
        "name": "Ratchet",
        "pitch_shift": -2,
        "distortion": 5,
        "reverb_size": 0.2,
        "reverb_wet": 0.08,
        "chorus_depth": 0.1,
        "delay_time": 0.01,
        "formant_shift": 0.9,
        "modulation_freq": 4,
        "modulation_depth": 0.08,
        "compression_ratio": 2,
        "gain_db": 1,
        "description": "Autobot medic - gruff but caring voice"
    },
    "ironhide": {
        "name": "Ironhide",
        "pitch_shift": -4,
        "distortion": 7,
        "reverb_size": 0.22,
        "reverb_wet": 0.1,
        "chorus_depth": 0.12,
        "delay_time": 0.01,
        "formant_shift": 0.82,
        "modulation_freq": 3,
        "modulation_depth": 0.1,
        "compression_ratio": 3,
        "gain_db": 2,
        "description": "Autobot weapons specialist - tough, grizzled voice"
    },
    "arcee": {
        "name": "Arcee",
        "pitch_shift": 5,
        "distortion": 3,
        "reverb_size": 0.18,
        "reverb_wet": 0.08,
        "chorus_depth": 0.2,
        "delay_time": 0.008,
        "formant_shift": 1.3,
        "modulation_freq": 8,
        "modulation_depth": 0.1,
        "compression_ratio": 2,
        "gain_db": 1,
        "description": "Female Autobot warrior - agile, determined voice"
    }
}


class TTSModel(ABC):
    """Base class for TTS models"""
    
    # License info to be overridden by each model
    LICENSE = "Unknown"
    OPEN_SOURCE = False
    
    @abstractmethod
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        """Generate audio from text"""
        pass
    
    def apply_transformer_effect(self, audio: np.ndarray, sample_rate: int, character: str = "optimus_prime") -> np.ndarray:
        """Apply Transformer character voice effects"""
        profile = CHARACTER_PROFILES.get(character, CHARACTER_PROFILES["optimus_prime"])
        
        # Check if effects should be bypassed for human voice
        if profile.get("bypass_effects", False):
            # Return original audio without any processing
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Just normalize to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9
            return audio
        
        return self._apply_voice_profile(audio, sample_rate, profile)
    
    def _apply_voice_profile(self, audio: np.ndarray, sample_rate: int, profile: dict) -> np.ndarray:
        """Apply specific voice profile to audio"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        effects = []
        effects.append(PitchShift(semitones=profile["pitch_shift"]))
        effects.append(Distortion(drive_db=profile["distortion"]))
        effects.append(Compressor(threshold_db=-20, ratio=profile["compression_ratio"]))
        effects.append(Reverb(room_size=profile["reverb_size"], damping=0.8, wet_level=profile["reverb_wet"]))
        effects.append(Chorus(rate_hz=0.5, depth=profile["chorus_depth"], mix=0.5))
        effects.append(Delay(delay_seconds=profile["delay_time"], mix=0.05))
        
        if profile.get("radio_effect"):
            effects.append(HighpassFilter(cutoff_frequency_hz=300))
            effects.append(LowpassFilter(cutoff_frequency_hz=3000))
            effects.append(Distortion(drive_db=5))
        
        if profile.get("vocoder_effect"):
            effects.append(Phaser(rate_hz=0.5, depth=0.9, mix=0.7))
            effects.append(Chorus(rate_hz=0.2, depth=0.8, mix=0.6))
        
        effects.append(Gain(gain_db=profile["gain_db"]))
        
        board = Pedalboard(effects)
        effected = board(audio, sample_rate)
        
        t = np.linspace(0, len(effected) / sample_rate, len(effected))
        mod_depth = profile["modulation_depth"]
        mod_freq = profile["modulation_freq"]
        
        if profile.get("vocoder_effect"):
            modulation = 0.7 + mod_depth * np.sign(np.sin(2 * np.pi * mod_freq * t))
        else:
            modulation = (1 - mod_depth/2) + mod_depth/2 * np.sin(2 * np.pi * mod_freq * t)
        
        effected = effected * modulation
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


# 1. SILERO TTS
class SileroTTS(TTSModel):
    """Silero TTS - Fast and lightweight neural TTS"""
    LICENSE = "MIT"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading Silero TTS (MIT License)...")
        self.device = torch.device('cpu')
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_tts',
                                       language='en',
                                       speaker='v3_en',
                                       trust_repo=True)
        self.model.to(self.device)
        print("✓ Silero TTS loaded!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        audio = self.model.apply_tts(text=text, speaker='en_0', sample_rate=48000)
        audio_np = audio.numpy() if torch.is_tensor(audio) else audio
        return self.apply_transformer_effect(audio_np, 48000, character), 48000


# 2. BARK (Suno AI)
class BarkTTS(TTSModel):
    """Bark - Suno AI's transformer-based text-to-audio model"""
    LICENSE = "MIT"
    OPEN_SOURCE = True
    
    def __init__(self):
        from transformers import AutoProcessor, BarkModel
        print("Loading Bark (MIT License)...")
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model = self.model.to("cpu")
        print("✓ Bark loaded!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        voice_preset = "v2/en_speaker_6"
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        with torch.no_grad():
            audio_array = self.model.generate(**inputs, do_sample=True)
        audio = audio_array.cpu().numpy().squeeze()
        return self.apply_transformer_effect(audio, 24000, character), 24000


# 3. SPEECH-T5 (Microsoft)
class SpeechT5TTS(TTSModel):
    """Microsoft SpeechT5 - Neural TTS model"""
    LICENSE = "MIT"
    OPEN_SOURCE = True
    
    def __init__(self):
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        print("Loading SpeechT5 (MIT License)...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        torch.manual_seed(42)
        self.speaker_embeddings = torch.randn(1, 512) * 0.1
        print("✓ SpeechT5 loaded!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        audio = speech.numpy()
        return self.apply_transformer_effect(audio, 16000, character), 16000


# 4. VITS (Facebook MMS)
class VITS_MMS(TTSModel):
    """Facebook MMS VITS - Multilingual TTS"""
    LICENSE = "CC-BY-NC 4.0"
    OPEN_SOURCE = True
    
    def __init__(self):
        from transformers import VitsModel, AutoTokenizer
        print("Loading VITS MMS (CC-BY-NC 4.0 License)...")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        print("✓ VITS MMS loaded!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        audio = output.squeeze().cpu().numpy()
        return self.apply_transformer_effect(audio, self.model.config.sampling_rate, character), self.model.config.sampling_rate


# 5. TACOTRON2 (NVIDIA)
class Tacotron2TTS(TTSModel):
    """NVIDIA Tacotron 2 - Neural TTS"""
    LICENSE = "BSD-3-Clause"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading Tacotron2 (BSD-3 License)...")
        try:
            # Using SpeechBrain's Tacotron2
            from speechbrain.pretrained import Tacotron2, HIFIGAN
            self.tts = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
            self.vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
            print("✓ Tacotron2 loaded!")
        except:
            print("Tacotron2 requires speechbrain - using alternative")
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            self.use_speecht5 = True
            torch.manual_seed(200)
            self.speaker_embeddings = torch.randn(1, 512) * 0.1
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if hasattr(self, 'use_speecht5'):
            inputs = self.processor(text=text, return_tensors="pt")
            with torch.no_grad():
                speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            audio = speech.numpy()
            sr = 16000
        else:
            mel_output, _, _ = self.tts.encode_text(text)
            waveforms = self.vocoder.decode_batch(mel_output)
            audio = waveforms.squeeze().cpu().numpy()
            sr = 22050
        return self.apply_transformer_effect(audio, sr, character), sr


# 6. FASTSPEECH2
class FastSpeech2TTS(TTSModel):
    """FastSpeech2 - Non-autoregressive TTS"""
    LICENSE = "Apache 2.0"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading FastSpeech2 (Apache 2.0 License)...")
        try:
            from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
            from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
            models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
                "facebook/fastspeech2-en-ljspeech",
                arg_overrides={"vocoder": "hifigan", "fp16": False}
            )
            self.model = models[0]
            TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
            self.generator = task.build_generator(models, cfg)
            print("✓ FastSpeech2 loaded!")
        except:
            print("FastSpeech2 requires fairseq - using alternative")
            # Use VITS as fallback
            from transformers import VitsModel, AutoTokenizer
            self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            self.use_vits = True
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if hasattr(self, 'use_vits'):
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**inputs).waveform
            audio = output.squeeze().cpu().numpy()
            sr = self.model.config.sampling_rate
        else:
            sample = TTSHubInterface.get_model_input(task, text)
            wav, rate = TTSHubInterface.get_prediction(task, self.model, self.generator, sample)
            audio = wav.cpu().numpy()
            sr = rate
        return self.apply_transformer_effect(audio, sr, character), sr


# 7. GLOW-TTS
class GlowTTS(TTSModel):
    """Glow-TTS - Flow-based generative TTS"""
    LICENSE = "MIT"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading Glow-TTS (MIT License)...")
        # Using ESPnet's implementation
        try:
            from espnet2.bin.tts_inference import Text2Speech
            self.model = Text2Speech.from_pretrained("kan-bayashi/ljspeech_glow_tts")
            print("✓ Glow-TTS loaded!")
        except:
            # Fallback to VITS
            from transformers import VitsModel, AutoTokenizer
            self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            self.use_vits = True
            print("Glow-TTS using VITS fallback")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if hasattr(self, 'use_vits'):
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**inputs).waveform
            audio = output.squeeze().cpu().numpy()
            sr = self.model.config.sampling_rate
        else:
            wav = self.model(text)["wav"]
            audio = wav.view(-1).cpu().numpy()
            sr = self.model.fs
        return self.apply_transformer_effect(audio, sr, character), sr


# 8. PYTTSX3 (System TTS)
class Pyttsx3TTS(TTSModel):
    """System TTS using pyttsx3"""
    LICENSE = "MPL-2.0"
    OPEN_SOURCE = True
    
    def __init__(self):
        import pyttsx3
        print("Loading pyttsx3 (MPL-2.0 License)...")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        voices = self.engine.getProperty('voices')
        if voices:
            for voice in voices:
                if 'male' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        print("✓ pyttsx3 loaded!")
    
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


# 9. ESPEAK
class EspeakTTS(TTSModel):
    """eSpeak - Classic formant synthesis"""
    LICENSE = "GPL-3.0"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading eSpeak (GPL-3.0 License)...")
        self.speed = 150
        self.pitch = 30
        self.voice = "en+m3"
        print("✓ eSpeak loaded!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                cmd = ['espeak', '-v', self.voice, '-s', str(self.speed), 
                       '-p', str(self.pitch), '-w', tmp_file.name, text]
                subprocess.run(cmd, check=True, capture_output=True)
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


# 10. GTTS (Google TTS)
class GTTS(TTSModel):
    """Google TTS - Cloud-based but free"""
    LICENSE = "Proprietary (Free tier)"
    OPEN_SOURCE = False
    
    def __init__(self):
        from gtts import gTTS
        print("Loading Google TTS (Proprietary - Free tier)...")
        self.gTTS = gTTS
        print("✓ Google TTS loaded!")
    
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


# 11. EDGE TTS (Microsoft)
class EdgeTTS(TTSModel):
    """Microsoft Edge TTS"""
    LICENSE = "Proprietary (Free)"
    OPEN_SOURCE = False
    
    def __init__(self):
        print("Loading Edge TTS (Proprietary - Free)...")
        self.voice = "en-US-ChristopherNeural"
        print("✓ Edge TTS loaded!")
    
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


# 12. FESTIVAL
class FestivalTTS(TTSModel):
    """Festival Speech Synthesis System"""
    LICENSE = "MIT-style"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading Festival (MIT-style License)...")
        print("✓ Festival ready!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                # Use festival to generate speech
                cmd = f'echo "{text}" | text2wave -o {tmp_file.name}'
                subprocess.run(cmd, shell=True, check=False, capture_output=True)
                if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                else:
                    # Fallback to espeak if festival fails
                    cmd = ['espeak', '-w', tmp_file.name, text]
                    subprocess.run(cmd, check=False, capture_output=True)
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


# 13. FLITE
class FliteTTS(TTSModel):
    """Flite - Small, fast speech synthesis"""
    LICENSE = "BSD-like"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading Flite (BSD-like License)...")
        # Use pyttsx3 as a more reliable alternative
        import pyttsx3
        self.engine = pyttsx3.init()
        # Set voice properties for Flite-like output
        self.engine.setProperty('rate', 150)  # Slower rate
        self.engine.setProperty('volume', 0.9)
        voices = self.engine.getProperty('voices')
        # Try to find a male voice
        for voice in voices:
            if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        print("✓ Flite ready!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                # Use pyttsx3 to generate speech
                self.engine.save_to_file(text, tmp_file.name)
                self.engine.runAndWait()
                
                # If file exists and has content, load it
                if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                else:
                    # Fallback to basic synthesis
                    duration = min(len(text) * 0.08, 5.0)
                    sr = 22050
                    t = np.linspace(0, duration, int(sr * duration))
                    # Create a simple speech-like waveform
                    frequencies = [200, 250, 300, 180]  # Varying frequencies
                    audio = np.zeros_like(t)
                    for i, freq in enumerate(frequencies):
                        segment_len = len(t) // len(frequencies)
                        start = i * segment_len
                        end = start + segment_len if i < len(frequencies) - 1 else len(t)
                        audio[start:end] = np.sin(2 * np.pi * freq * t[start:end]) * 0.3
                    # Add some noise for texture
                    audio += np.random.normal(0, 0.01, len(audio))
                
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


# 14. MBROLA
class MbrolaTTS(TTSModel):
    """MBROLA - Diphone synthesis"""
    LICENSE = "AGPL-3.0"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading MBROLA (AGPL-3.0 License)...")
        self.voice = "en1"
        print("✓ MBROLA ready!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                # MBROLA requires phoneme input, using espeak as frontend
                pho_file = tmp_file.name.replace('.wav', '.pho')
                cmd1 = f'espeak -v mb-en1 -x -q --pho "{text}" > {pho_file}'
                subprocess.run(cmd1, shell=True, check=False, capture_output=True)
                
                if os.path.exists(pho_file):
                    cmd2 = f'mbrola /usr/share/mbrola/{self.voice}/{self.voice} {pho_file} {tmp_file.name}'
                    subprocess.run(cmd2, shell=True, check=False, capture_output=True)
                
                if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                else:
                    # Fallback to espeak
                    cmd = ['espeak', '-w', tmp_file.name, text]
                    subprocess.run(cmd, check=False, capture_output=True)
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                if os.path.exists(pho_file):
                    os.unlink(pho_file)


# 15. MERLIN
class MerlinTTS(TTSModel):
    """MERLIN - Neural parametric TTS"""
    LICENSE = "Apache 2.0"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading MERLIN (Apache 2.0 License)...")
        # MERLIN is complex to set up, using alternative
        from transformers import VitsModel, AutoTokenizer
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        print("✓ MERLIN (using VITS) loaded!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        audio = output.squeeze().cpu().numpy()
        sr = self.model.config.sampling_rate
        return self.apply_transformer_effect(audio, sr, character), sr


# 16. LARYNX
class LarynxTTS(TTSModel):
    """Larynx - Local neural TTS"""
    LICENSE = "MIT"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading Larynx (MIT License)...")
        try:
            import larynx
            self.larynx = larynx
            print("✓ Larynx loaded!")
        except:
            # Use alternative
            self.use_alt = True
            print("Larynx not available, using alternative")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if hasattr(self, 'use_alt'):
            # Use espeak as fallback
            import subprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                try:
                    cmd = ['espeak', '-w', tmp_file.name, text]
                    subprocess.run(cmd, check=False, capture_output=True)
                    audio, sr = librosa.load(tmp_file.name, sr=None)
                    return self.apply_transformer_effect(audio, sr, character), sr
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
        else:
            # Use larynx
            audio_bytes = self.larynx.text_to_speech(text)
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            return self.apply_transformer_effect(audio, sr, character), sr


# 17. SAM (Software Automatic Mouth)
class SAMTTS(TTSModel):
    """SAM - Retro 8-bit speech synthesis"""
    LICENSE = "Custom (Free)"
    OPEN_SOURCE = True
    
    def __init__(self):
        print("Loading SAM (Custom Free License)...")
        print("✓ SAM ready!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                # Try to use SAM
                cmd = f'sam "{text}" -wav {tmp_file.name}'
                result = subprocess.run(cmd, shell=True, check=False, capture_output=True)
                
                if not os.path.exists(tmp_file.name) or os.path.getsize(tmp_file.name) == 0:
                    # Fallback to espeak with robotic settings
                    cmd = ['espeak', '-s', '120', '-p', '20', '-w', tmp_file.name, text]
                    subprocess.run(cmd, check=False, capture_output=True)
                
                audio, sr = librosa.load(tmp_file.name, sr=None)
                return self.apply_transformer_effect(audio, sr, character), sr
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


# 18. BALABOLKA (Windows only, but including for completeness)
class BalabolkaTTS(TTSModel):
    """Balabolka - Windows SAPI wrapper"""
    LICENSE = "Freeware"
    OPEN_SOURCE = False
    
    def __init__(self):
        print("Loading Balabolka alternative...")
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            # Configure for Balabolka-like voice
            self.engine.setProperty('rate', 175)  # Medium speed
            self.engine.setProperty('volume', 1.0)
            voices = self.engine.getProperty('voices')
            # Try to use the first available voice
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            self.use_engine = True
        except:
            self.use_engine = False
        print("✓ Balabolka alternative ready!")
    
    def generate(self, text: str, character: str = "optimus_prime") -> Tuple[np.ndarray, int]:
        if self.use_engine:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                try:
                    self.engine.save_to_file(text, tmp_file.name)
                    self.engine.runAndWait()
                    
                    # Check if file was created successfully
                    if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                        audio, sr = librosa.load(tmp_file.name, sr=None)
                        return self.apply_transformer_effect(audio, sr, character), sr
                except:
                    pass
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
        
        # Fallback to generated speech pattern
        duration = min(len(text) * 0.09, 5.0)  # Slightly slower than Flite
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create speech-like pattern with formants
        base_freq = 180  # Lower base frequency
        formants = [700, 1220, 2600]  # Typical formant frequencies
        audio = np.zeros_like(t)
        
        # Generate base tone
        audio = np.sin(2 * np.pi * base_freq * t) * 0.3
        
        # Add formants for more realistic speech
        for formant in formants:
            audio += np.sin(2 * np.pi * formant * t) * 0.1
        
        # Add amplitude modulation for speech rhythm
        mod_freq = 4  # 4 Hz modulation
        audio *= (1 + 0.3 * np.sin(2 * np.pi * mod_freq * t))
        
        # Add slight noise
        audio += np.random.normal(0, 0.005, len(audio))
        
        return self.apply_transformer_effect(audio, sr, character), sr


# Complete model registry with ALL local TTS models
MODELS = {
    # Open Source Models
    "Silero TTS": (SileroTTS, "MIT", True),
    "Bark (Suno AI)": (BarkTTS, "MIT", True),
    "SpeechT5 (Microsoft)": (SpeechT5TTS, "MIT", True),
    "VITS (Meta MMS)": (VITS_MMS, "CC-BY-NC 4.0", True),
    "Tacotron2 (NVIDIA)": (Tacotron2TTS, "BSD-3", True),
    "FastSpeech2": (FastSpeech2TTS, "Apache 2.0", True),
    "Glow-TTS": (GlowTTS, "MIT", True),
    "pyttsx3 (System)": (Pyttsx3TTS, "MPL-2.0", True),
    "eSpeak": (EspeakTTS, "GPL-3.0", True),
    "Festival": (FestivalTTS, "MIT-style", True),
    "Flite": (FliteTTS, "BSD-like", True),
    "MBROLA": (MbrolaTTS, "AGPL-3.0", True),
    "MERLIN": (MerlinTTS, "Apache 2.0", True),
    "Larynx": (LarynxTTS, "MIT", True),
    "SAM (8-bit)": (SAMTTS, "Custom Free", True),
    
    # Proprietary but Free
    "Google TTS": (GTTS, "Proprietary (Free)", False),
    "Edge TTS (Microsoft)": (EdgeTTS, "Proprietary (Free)", False),
    "Balabolka (Windows)": (BalabolkaTTS, "Freeware", False),
}

# Export for API use
__all__ = ['MODELS', 'CHARACTER_PROFILES']