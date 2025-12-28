"""Speech-to-text functionality using OpenAI's Whisper."""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
import whisper
from loguru import logger

from ..config import config

class WhisperSTT:
    """Speech-to-text using OpenAI's Whisper model."""
    
    def __init__(self, model_size: str = None, device: str = None):
        """Initialize the Whisper model.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run the model on (cuda, cpu, or auto)
        """
        self.model_size = model_size or config.get("stt.model_size", "base")
        self.device = device or self._get_device()
        self.model = None
        self._load_model()
    
    def _get_device(self) -> str:
        """Get the appropriate device for Whisper."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio to text.
        
        Args:
            audio: Path to audio file or audio data as numpy array or torch.Tensor
            language: Language code (e.g., 'en', 'es', 'fr')
            **kwargs: Additional arguments to pass to whisper.transcribe()
            
        Returns:
            Dictionary containing the transcription and metadata
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Set default parameters if not provided
            options = {
                "language": language or config.get("stt.language", "en"),
                "fp16": False if self.device == "cpu" else True,
                **kwargs
            }
            
            # Transcribe the audio
            result = self.model.transcribe(audio, **options)
            
            logger.debug(f"Transcription successful: {result['text'][:100]}...")
            return {
                "text": result["text"].strip(),
                "language": result.get("language"),
                "segments": result.get("segments", []),
                "language_probability": result.get("language_probability"
            )}
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def transcribe_file(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file.
        
        Args:
            file_path: Path to the audio file
            **kwargs: Additional arguments to pass to transcribe()
            
        Returns:
            Dictionary containing the transcription and metadata
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        return self.transcribe(str(file_path), **kwargs)

# Singleton instance for easy import
whisper_stt = WhisperSTT()
