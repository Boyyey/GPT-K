"""
Text-to-Speech Module

Provides TTS functionality using Piper TTS for offline, high-quality speech synthesis.
"""
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union, BinaryIO

import numpy as np
import onnxruntime as ort
from loguru import logger

# Add piper-phonemize to requirements
# Add piper_phonemize>=1.0.0 to requirements.txt

class TTS:
    """Offline Text-to-Speech using Piper TTS."""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        use_cuda: bool = False,
        speaker_id: int = 0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        """Initialize the TTS engine.
        
        Args:
            model_path: Path to the ONNX model file
            config_path: Path to the model config JSON (optional)
            use_cuda: Whether to use CUDA for inference
            speaker_id: Speaker ID for multi-speaker models
            length_scale: Control speech speed (lower = faster)
            noise_scale: Control voice stability (0.0-1.0)
            noise_w: Control voice variation (0.0-1.0)
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        
        # Initialize ONNX runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=self.session_options,
                providers=providers
            )
            logger.info(f"Loaded TTS model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
        
        # Load phonemizer
        self._init_phonemizer()
        
        # Audio parameters
        self.sample_rate = 22050  # Default, can be overridden by model
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _init_phonemizer(self):
        """Initialize the phonemizer for text processing."""
        try:
            from piper_phonemize import phonemize_espeak, phonemes_to_ids, get_codepoints_map, get_phoneme_id_map
            self.phonemize = phonemize_espeak
            self.phonemes_to_ids = phonemes_to_ids
            self.get_codepoints_map = get_codepoints_map
            self.get_phoneme_id_map = get_phoneme_id_map
            logger.debug("Initialized phonemizer")
        except ImportError:
            logger.warning("piper-phonemize not found, falling back to basic text processing")
            self.phonemize = lambda text, **kwargs: [text]
            self.phonemes_to_ids = lambda phonemes, **kwargs: [0] * len(phonemes)
    
    def synthesize(
        self,
        text: str,
        output_file: Optional[Union[str, Path, BinaryIO]] = None,
        format: str = 'wav',
        **kwargs
    ) -> Optional[np.ndarray]:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            output_file: Optional file path or file-like object to save audio
            format: Output format ('wav', 'raw', or 'numpy')
            **kwargs: Override default parameters (length_scale, noise_scale, noise_w)
            
        Returns:
            Audio data as numpy array if output_file is None, else None
        """
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None
        
        # Override defaults with kwargs
        length_scale = kwargs.get('length_scale', self.length_scale)
        noise_scale = kwargs.get('noise_scale', self.noise_scale)
        noise_w = kwargs.get('noise_w', self.noise_w)
        
        try:
            # Preprocess text
            phonemes = self._preprocess_text(text)
            
            # Generate audio
            with self._lock:
                audio = self._synthesize_phonemes(
                    phonemes,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w=noise_w
                )
            
            # Convert to int16 for WAV output
            if format.lower() == 'wav':
                audio = (audio * 32767).astype(np.int16)
            
            # Save to file if requested
            if output_file is not None:
                self._save_audio(audio, output_file, format)
                return None
                
            return audio
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS."""
        # Basic text cleaning
        text = text.strip()
        if not text:
            return ""
            
        # Simple implementation - in a real app, you'd use a proper text normalizer
        text = text.replace('"', '').replace('\n', ' ')
        return text
    
    def _synthesize_phonemes(
        self,
        text: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> np.ndarray:
        """Synthesize speech from preprocessed text."""
        # Tokenize text into phonemes
        phonemes = self.phonemize(text)
        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        # Prepare model inputs
        phoneme_ids = np.array([phoneme_ids], dtype=np.int64)
        phoneme_lengths = np.array([len(phoneme_ids[0])], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32
        )
        
        # Run inference
        ort_inputs = {
            'input': phoneme_ids,
            'input_lengths': phoneme_lengths,
            'scales': scales,
            'sid': np.array([self.speaker_id], dtype=np.int64)
        }
        
        audio = self.session.run(None, ort_inputs)[0].squeeze()
        
        # Normalize audio
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _save_audio(
        self,
        audio: np.ndarray,
        output: Union[str, Path, BinaryIO],
        format: str = 'wav'
    ) -> None:
        """Save audio to file."""
        import wave
        import struct
        
        if format.lower() == 'wav':
            if isinstance(output, (str, Path)):
                with wave.open(str(output), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio.tobytes())
            else:
                with wave.open(output, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio.tobytes())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def stream_speech(self, text_generator, **kwargs):
        """Stream synthesized speech from a text generator."""
        for text_chunk in text_generator:
            if not text_chunk.strip():
                continue
                
            audio = self.synthesize(text_chunk, format='numpy', **kwargs)
            if audio is not None:
                yield audio
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            del self.session
