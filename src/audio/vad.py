"""
Voice Activity Detection (VAD) Module

Detects voice activity in audio streams using WebRTC VAD.
"""
import collections
import numpy as np
from typing import Deque, Optional, Tuple

import webrtcvad
from loguru import logger

class VAD:
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        padding_duration_ms: int = 300,
        aggressiveness: int = 3,
        min_speech_duration: float = 0.5,
        min_silence_duration: float = 0.3,
    ):
        """Initialize the VAD.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Duration of each audio frame in milliseconds
            padding_duration_ms: Duration to pad speech segments (ms)
            aggressiveness: VAD aggressiveness (0-3, 3 is most aggressive)
            min_speech_duration: Minimum duration of speech to consider (seconds)
            min_silence_duration: Minimum duration of silence to consider (seconds)
        """
        if not 0 <= aggressiveness <= 3:
            raise ValueError("Aggressiveness must be between 0 and 3")
            
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = (sample_rate * frame_duration_ms) // 1000
        self.padding_duration_ms = padding_duration_ms
        self.padding_frames = padding_duration_ms // frame_duration_ms
        self.min_speech_frames = int(min_speech_duration * 1000 / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration * 1000 / frame_duration_ms)
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self.audio_buffer: Deque[np.ndarray] = collections.deque(maxlen=1000)
        self.speech_buffer: Deque[bool] = collections.deque(maxlen=1000)
        self.speech_history: Deque[bool] = collections.deque(maxlen=100)
        
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
    
    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Process a single audio frame for voice activity.
        
        Args:
            audio_frame: Mono audio frame as numpy array
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        if len(audio_frame) != self.frame_size:
            logger.warning(f"Unexpected frame size: {len(audio_frame)} != {self.frame_size}")
            return False
        
        # Convert to 16-bit PCM for VAD
        if audio_frame.dtype != np.int16:
            audio_frame = (audio_frame * 32767).astype(np.int16)
        
        # Check if frame contains speech
        is_speech = self.vad.is_speech(
            audio_frame.tobytes(),
            sample_rate=self.sample_rate
        )
        
        # Update buffers and state
        self.audio_buffer.append(audio_frame)
        self.speech_buffer.append(is_speech)
        self.speech_history.append(is_speech)
        
        # Update state machine
        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1
        else:
            self.silence_frames += 1
            self.speech_frames = max(0, self.speech_frames - 1)
        
        # State transitions
        if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
            self.is_speaking = True
            logger.debug("Speech start detected")
        elif self.is_speaking and self.silence_frames >= self.min_silence_frames:
            self.is_speaking = False
            logger.debug("Speech end detected")
        
        return self.is_speaking
    
    def get_audio_chunk(self) -> Optional[Tuple[np.ndarray, bool]]:
        """Get the next chunk of audio with speech/silence information.
        
        Returns:
            Tuple of (audio_frame, is_speech) or None if buffer is empty
        """
        if not self.audio_buffer or not self.speech_buffer:
            return None
            
        return self.audio_buffer.popleft(), self.speech_buffer.popleft()
    
    def get_speech_segment(self) -> Optional[np.ndarray]:
        """Get a continuous segment of speech audio.
        
        Returns:
            Concatenated audio frames containing speech or None if no speech
        """
        if not self.is_speaking or len(self.audio_buffer) < self.min_speech_frames:
            return None
        
        # Find speech segments
        speech_segments = []
        current_segment = []
        
        for frame, is_speech in zip(self.audio_buffer, self.speech_buffer):
            if is_speech:
                current_segment.append(frame)
            elif current_segment:
                if len(current_segment) >= self.min_speech_frames:
                    speech_segments.append(np.concatenate(current_segment))
                current_segment = []
        
        # Add the last segment if it ends with speech
        if current_segment and len(current_segment) >= self.min_speech_frames:
            speech_segments.append(np.concatenate(current_segment))
        
        if not speech_segments:
            return None
        
        # Concatenate all speech segments with padding
        return np.concatenate(speech_segments)
    
    def reset(self):
        """Reset the VAD state."""
        self.audio_buffer.clear()
        self.speech_buffer.clear()
        self.speech_history.clear()
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
