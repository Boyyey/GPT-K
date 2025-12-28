"""
LLM Engine

Core LLM inference engine using llama.cpp.
"""
import os
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Generator

import llama_cpp
from loguru import logger

class LLMConfig:
    """Configuration for the LLM engine."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: int = 0,  # 0 = use all available
        n_gpu_layers: int = 0,  # 0 = CPU only
        seed: int = -1,  # -1 for random
        f16_kv: bool = True,
        use_mlock: bool = False,
        use_mmap: bool = True,
        embedding: bool = False,
        last_n_tokens_size: int = 64,
        logits_all: bool = False,
        vocab_only: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize LLM configuration."""
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads or max(1, os.cpu_count() // 2)
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.f16_kv = f16_kv
        self.use_mlock = use_mlock
        self.use_mmap = use_mmap
        self.embedding = embedding
        self.last_n_tokens_size = last_n_tokens_size
        self.logits_all = logits_all
        self.vocab_only = vocab_only
        self.verbose = verbose
        
        # Update with any additional parameters
        self.__dict__.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}

@dataclass
class LLMResponse:
    """Response from the LLM engine."""
    text: str = ""
    tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_taken: float = 0.0
    model: str = ""
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'text': self.text,
            'tokens': self.tokens,
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'time_taken': self.time_taken,
            'model': self.model,
            'finish_reason': self.finish_reason,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the response."""
        return self.text

class LLMEngine:
    """LLM engine for local inference using llama.cpp."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM engine."""
        self.config = config
        self.model = None
        self.ctx = None
        self._lock = threading.RLock()
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model."""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        try:
            start_time = time.time()
            
            # Initialize model parameters
            model_params = {
                'model_path': self.config.model_path,
                'n_ctx': self.config.n_ctx,
                'n_batch': self.config.n_batch,
                'n_threads': self.config.n_threads,
                'n_gpu_layers': self.config.n_gpu_layers,
                'seed': self.config.seed,
                'f16_kv': self.config.f16_kv,
                'use_mlock': self.config.use_mlock,
                'use_mmap': self.config.use_mmap,
                'embedding': self.config.embedding,
                'last_n_tokens_size': self.config.last_n_tokens_size,
                'logits_all': self.config.logits_all,
                'vocab_only': self.config.vocab_only,
                'verbose': self.config.verbose,
            }
            
            # Initialize the model
            with self._lock:
                self.model = llama_cpp.Llama(**model_params)
                self.ctx = self.model.context
                
                # Warm up the model with a small inference
                self.model("Hello, world!", max_tokens=1)
                
            load_time = time.time() - start_time
            logger.info(f"Loaded model in {load_time:.2f}s: {self.config.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[LLMResponse, None, None]]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2, higher = more creative)
            top_p: Nucleus sampling (0-1)
            top_k: Top-k sampling (0 = disabled)
            repeat_penalty: Penalty for repeating tokens (1.0 = no penalty)
            stop: Stop sequence(s)
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse or generator of LLMResponse chunks
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Convert stop to list if it's a string
        if isinstance(stop, str):
            stop = [stop]
        
        # Prepare generation parameters
        gen_params = {
            'max_tokens': max_tokens,
            'temperature': max(0.01, min(temperature, 2.0)),
            'top_p': max(0.0, min(top_p, 1.0)),
            'top_k': max(0, top_k),
            'repeat_penalty': max(0.0, repeat_penalty),
            'stop': stop,
            **kwargs
        }
        
        # For streaming responses
        if stream:
            return self._stream_generate(prompt, **gen_params)
        
        # For non-streaming responses
        start_time = time.time()
        
        try:
            with self._lock:
                # Tokenize the prompt to count tokens
                prompt_tokens = self.model.tokenize(prompt.encode('utf-8'))
                
                # Generate the response
                response = self.model(
                    prompt=prompt,
                    **{k: v for k, v in gen_params.items() 
                       if k != 'stop' and not k.startswith('_')}
                )
                
                # Extract the generated text
                if isinstance(response, dict):
                    # Newer API returns a dict
                    text = response.get('choices', [{}])[0].get('text', '')
                    finish_reason = response.get('choices', [{}])[0].get('finish_reason', '')
                    usage = response.get('usage', {})
                    completion_tokens = usage.get('completion_tokens', 0)
                    prompt_tokens_count = usage.get('prompt_tokens', len(prompt_tokens))
                    total_tokens = usage.get('total_tokens', prompt_tokens_count + completion_tokens)
                else:
                    # Fallback for older API
                    text = response
                    finish_reason = "length" if len(text.split()) >= max_tokens else "stop"
                    completion_tokens = len(self.model.tokenize(text.encode('utf-8')))
                    prompt_tokens_count = len(prompt_tokens)
                    total_tokens = prompt_tokens_count + completion_tokens
                
                # Create response object
                result = LLMResponse(
                    text=text,
                    tokens=total_tokens,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens_count,
                    completion_tokens=completion_tokens,
                    time_taken=time.time() - start_time,
                    model=os.path.basename(self.config.model_path),
                    finish_reason=finish_reason,
                    metadata={
                        'temperature': temperature,
                        'top_p': top_p,
                        'top_k': top_k,
                        'repeat_penalty': repeat_penalty,
                        'stop_sequence': stop,
                    }
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _stream_generate(self, prompt: str, **kwargs) -> Generator[LLMResponse, None, None]:
        """Generate text with streaming support."""
        start_time = time.time()
        buffer = ""
        
        try:
            with self._lock:
                # Tokenize the prompt to count tokens
                prompt_tokens = self.model.tokenize(prompt.encode('utf-8'))
                prompt_tokens_count = len(prompt_tokens)
                
                # Create a streaming response
                stream = self.model(
                    prompt=prompt,
                    stream=True,
                    **{k: v for k, v in kwargs.items() 
                       if k != 'stop' and not k.startswith('_')}
                )
                
                # Process the stream
                completion_tokens = 0
                for chunk in stream:
                    if not chunk:
                        continue
                        
                    # Extract the text delta
                    if isinstance(chunk, dict):
                        # Newer API returns a dict
                        delta = chunk.get('choices', [{}])[0].get('text', '')
                        finish_reason = chunk.get('choices', [{}])[0].get('finish_reason', '')
                    else:
                        # Fallback for older API
                        delta = chunk
                        finish_reason = ""
                    
                    if not delta:
                        continue
                    
                    # Update the buffer and token count
                    buffer += delta
                    completion_tokens += 1
                    
                    # Create a response object for this chunk
                    result = LLMResponse(
                        text=delta,
                        tokens=prompt_tokens_count + completion_tokens,
                        total_tokens=prompt_tokens_count + completion_tokens,
                        prompt_tokens=prompt_tokens_count,
                        completion_tokens=completion_tokens,
                        time_taken=time.time() - start_time,
                        model=os.path.basename(self.config.model_path),
                        finish_reason=finish_reason,
                        metadata={
                            'chunk': True,
                            'finish_reason': finish_reason or None,
                        }
                    )
                    
                    yield result
                
                # Final response with the complete text
                final_response = LLMResponse(
                    text=buffer,
                    tokens=prompt_tokens_count + completion_tokens,
                    total_tokens=prompt_tokens_count + completion_tokens,
                    prompt_tokens=prompt_tokens_count,
                    completion_tokens=completion_tokens,
                    time_taken=time.time() - start_time,
                    model=os.path.basename(self.config.model_path),
                    finish_reason="stop",
                    metadata={
                        'chunk': False,
                        'finish_reason': 'stop',
                    }
                )
                
                yield final_response
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the input text."""
        if not self.model or not hasattr(self.model, 'embed'):
            raise NotImplementedError("Embedding not supported by this model")
        
        try:
            with self._lock:
                return self.model.embed(text)
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize the input text."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            with self._lock:
                return self.model.tokenize(text.encode('utf-8'))
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            with self._lock:
                return self.model.detokenize(tokens).decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            raise
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            self.ctx = None
