"""
Audio I/O Module

Handles microphone input and speaker output using sounddevice.
"""
import queue
import threading
import time
from typing import Optional, Callable, List

import numpy as np
import sounddevice as sd
from loguru import logger

class AudioIO:
    """Handles audio input and output operations."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        """Initialize the audio I/O.
        
        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            chunk_size: Number of frames per buffer
            input_device: Input device ID or None for default
            output_device: Output device ID or None for default
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.input_device = input_device
        self.output_device = output_device
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread: Optional[threading.Thread] = None
        self.callback = None
        
        # Test audio devices
        self._test_audio_devices()
    
    def _test_audio_devices(self):
        """Test audio devices and log available options."""
        try:
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"{i}: {device['name']} (Inputs: {device['max_input_channels']}, "
                          f"Outputs: {device['max_output_channels']})")
            
            # Test default devices
            sd.check_input_settings(device=self.input_device, samplerate=self.sample_rate, 
                                  channels=self.channels, dtype='float32')
            sd.check_output_settings(device=self.output_device, samplerate=self.sample_rate,
                                   channels=self.channels, dtype='float32')
        except Exception as e:
            logger.error(f"Audio device error: {e}")
            raise
    
    def start_recording(self, callback: Optional[Callable[[np.ndarray], None]] = None):
        """Start recording audio from the microphone.
        
        Args:
            callback: Function to call with audio chunks
        """
        if self.is_recording:
            logger.warning("Already recording")
            return
            
        self.callback = callback
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.audio_thread.start()
        logger.info("Started audio recording")
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        logger.info("Stopped audio recording")
    
    def _record_loop(self):
        """Main recording loop."""
        def audio_callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            if self.callback:
                self.callback(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=self.chunk_size,
                device=self.input_device,
                dtype='float32'
            ) as stream:
                while self.is_recording and stream.active:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.is_recording = False
    
    def play_audio(self, audio_data: np.ndarray, blocking: bool = True):
        """Play audio data.
        
        Args:
            audio_data: Audio data as numpy array
            blocking: If True, block until playback is finished
        """
        try:
            if len(audio_data) == 0:
                logger.warning("Empty audio data, nothing to play")
                return
                
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
            
            sd.play(
                audio_data,
                samplerate=self.sample_rate,
                device=self.output_device,
                blocking=blocking
            )
            
            if not blocking:
                return sd.get_stream()
                
        except Exception as e:
            logger.error(f"Playback error: {e}")
            raise
    
    def get_audio_devices(self) -> List[dict]:
        """Get a list of available audio devices.
        
        Returns:
            List of dictionaries with device information
        """
        return sd.query_devices()
    
    def set_input_device(self, device_id: int):
        """Set the input device by ID.
        
        Args:
            device_id: Device ID to use for input
        """
        devices = self.get_audio_devices()
        if 0 <= device_id < len(devices):
            self.input_device = device_id
            logger.info(f"Set input device to: {devices[device_id]['name']}")
        else:
            logger.warning(f"Invalid device ID: {device_id}")
    
    def set_output_device(self, device_id: int):
        """Set the output device by ID.
        
        Args:
            device_id: Device ID to use for output
        """
        devices = self.get_audio_devices()
        if 0 <= device_id < len(devices):
            self.output_device = device_id
            logger.info(f"Set output device to: {devices[device_id]['name']}")
        else:
            logger.warning(f"Invalid device ID: {device_id}")
