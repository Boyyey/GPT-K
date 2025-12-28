"""
Speech-to-Text Module

Provides offline speech recognition using Whisper.cpp.
"""
import os
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable

import whisper_cpp
from loguru import logger

class WhisperSTT:
    """Offline Speech-to-Text using Whisper.cpp."""
    
    def __init__(
        self,
        model_path: str,
        language: str = "en",
        translate: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        vad_threshold: float = 0.6,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -1.0,
        no_speech_threshold_long: float = 0.5,
        log_prob_threshold_long: float = -1.0,
        max_initial_ts: float = 1.0,
        length_penalty: float = -0.1,
        temperature_inc: float = 0.4,
        entropy_threshold: float = 2.4,
        log_prob_threshold_positive: float = 0.5,
        log_prob_threshold_negative: float = -1.0,
    ):
        """Initialize the Whisper STT engine.
        
        Args:
            model_path: Path to the Whisper GGML model file
            language: Language code (e.g., 'en', 'es', 'fr')
            translate: Whether to translate to English
            beam_size: Beam size for beam search
            best_of: Number of candidates when sampling with non-zero temperature
            temperature: Temperature for sampling
            vad_threshold: Voice activity detection threshold (0-1)
            no_speech_threshold: Threshold for considering a segment as silence
            log_prob_threshold: Log probability threshold for accepting tokens
            no_speech_threshold_long: No-speech threshold for long-form transcription
            log_prob_threshold_long: Log probability threshold for long-form
            max_initial_ts: Maximum initial timestamp in seconds
            length_penalty: Length penalty (negative for shorter, positive for longer)
            temperature_inc: Temperature increment for fallback
            entropy_threshold: Entropy threshold for fallback
            log_prob_threshold_positive: Log probability threshold for positive samples
            log_prob_threshold_negative: Log probability threshold for negative samples
        """
        self.model_path = Path(model_path)
        self.language = language
        self.translate = translate
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.vad_threshold = vad_threshold
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold_long = no_speech_threshold_long
        self.log_prob_threshold_long = log_prob_threshold_long
        self.max_initial_ts = max_initial_ts
        self.length_penalty = length_penalty
        self.temperature_inc = temperature_inc
        self.entropy_threshold = entropy_threshold
        self.log_prob_threshold_positive = log_prob_threshold_positive
        self.log_prob_threshold_negative = log_prob_threshold_negative
        
        # Thread safety
        self._lock = threading.Lock()
        self._initialized = False
        self._model = None
        
        # Initialize the model
        self._init_model()
    
    def _init_model(self):
        """Initialize the Whisper model."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        
        try:
            # Initialize Whisper model
            self._model = whisper_cpp.Whisper.from_pretrained(
                model_name_or_path=str(self._model_path.parent),
                model_file=self._model_path.name,
                download_root=None,
                in_memory=False,
            )
            
            # Set model parameters
            self._model.params.language = self.language
            self._model.params.translate = self.translate
            self._model.params.beam_size = self.beam_size
            self._model.params.best_of = self.best_of
            self._model.params.temperature = self.temperature
            self._model.params.vad_thold = self.vad_threshold
            self._model.params.no_speech_thold = self.no_speech_threshold
            self._model.params.logprob_thold = self.log_prob_threshold
            self._model.params.no_speech_thold_long = self.no_speech_threshold_long
            self._model.params.logprob_thold_long = self.log_prob_threshold_long
            self._model.params.max_initial_ts = self.max_initial_ts
            self._model.params.length_penalty = self.length_penalty
            self._model.params.temperature_inc = self.temperature_inc
            self._model.params.entropy_thold = self.entropy_threshold
            self._model.params.logprob_thold_positive = self.log_prob_threshold_positive
            self._model.params.logprob_thold_negative = self.log_prob_threshold_negative
            
            self._initialized = True
            logger.info(f"Initialized Whisper model: {self._model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise
    
    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        language: Optional[str] = None,
        translate: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array or path to audio file
            language: Override language setting
            translate: Override translate setting
            **kwargs: Additional parameters to override
            
        Returns:
            Dictionary containing transcription results
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Load audio if path is provided
            if isinstance(audio, (str, Path)):
                audio = self._load_audio(audio)
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0  # Convert from int16
            
            # Override parameters
            params = {
                'language': language if language is not None else self.language,
                'translate': translate if translate is not None else self.translate,
                **kwargs
            }
            
            # Transcribe
            with self._lock:
                result = self._model.transcribe(audio, **params)
            
            return {
                'text': result.get('text', '').strip(),
                'language': result.get('language', self.language),
                'segments': result.get('segments', []),
                'is_translated': result.get('is_translated', False),
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def stream_transcribe(
        self,
        audio_generator,
        language: Optional[str] = None,
        translate: Optional[bool] = None,
        chunk_size: int = 30,  # seconds
        **kwargs
    ):
        """Stream transcription for real-time audio.
        
        Args:
            audio_generator: Generator yielding audio chunks
            language: Override language setting
            translate: Override translate setting
            chunk_size: Size of audio chunks in seconds
            **kwargs: Additional parameters
            
        Yields:
            Transcription results for each chunk
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Model not initialized")
        
        # Override parameters
        params = {
            'language': language if language is not None else self.language,
            'translate': translate if translate is not None else self.translate,
            **kwargs
        }
        
        # Initialize streaming state
        buffer = []
        sample_rate = 16000  # Default, should match your audio source
        samples_per_chunk = chunk_size * sample_rate
        
        try:
            for audio_chunk in audio_generator:
                if audio_chunk is None:
                    continue
                
                # Add to buffer
                buffer.append(audio_chunk)
                
                # Process if buffer is large enough
                if len(buffer) >= samples_per_chunk:
                    # Concatenate chunks
                    audio = np.concatenate(buffer)
                    
                    # Transcribe
                    result = self.transcribe(audio, **params)
                    
                    # Clear buffer
                    buffer = []
                    
                    # Yield result
                    yield result
            
            # Process remaining audio
            if buffer:
                audio = np.concatenate(buffer)
                result = self.transcribe(audio, **params)
                yield result
                
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise
    
    def _load_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load audio file into numpy array."""
        import soundfile as sf
        
        try:
            audio, sample_rate = sf.read(file_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            return audio
            
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    def get_available_languages(self) -> List[str]:
        """Get list of supported language codes."""
        # Whisper supports these languages out of the box
        return [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr',
            'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi',
            'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no',
            'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk',
            'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk',
            'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw',
            'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc',
            'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo',
            'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl',
            'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'
        ]
    
    def set_language(self, language: str) -> None:
        """Set the language for transcription."""
        if language not in self.get_available_languages():
            logger.warning(f"Language not supported: {language}")
            return
            
        self.language = language
        if self._model is not None:
            self._model.params.language = language
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._initialized = False
