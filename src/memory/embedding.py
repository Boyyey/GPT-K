"""Embedding model for generating and managing vector embeddings."""
import os
import json
from typing import List, Dict, Any, Optional, Union

import numpy as np
from loguru import logger

# Try to import sentence_transformers, but make it optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class EmbeddingModel:
    """Handles text embedding generation and management."""
    
    # Default model configuration
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # Small but effective model
    DEFAULT_EMBEDDING_DIM = 384  # Dimension of the embeddings
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
            cache_dir: Directory to cache the model
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        
        # Initialize the model
        self._model = None
        self._embedding_dim = None
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence-transformers model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Get the embedding dimension
            test_embedding = self._model.encode("test", convert_to_numpy=True)
            self._embedding_dim = test_embedding.shape[0]
            
            logger.info(f"Loaded embedding model with dimension {self._embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        if self._embedding_dim is None:
            raise RuntimeError("Model not properly initialized")
        return self._embedding_dim
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            convert_to_numpy: Whether to convert output to numpy arrays
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings or list of numpy arrays
        """
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Handle single string input
            is_single = isinstance(texts, str)
            if is_single:
                texts = [texts]
            
            # Encode the texts
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
            
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_to_json(self, text: str) -> str:
        """Encode text and return the embedding as a JSON string."""
        embedding = self.encode(text, convert_to_numpy=True)
        return json.dumps(embedding.tolist())
    
    @staticmethod
    def json_to_embedding(embedding_json: str) -> np.ndarray:
        """Convert a JSON string back to a numpy array."""
        return np.array(json.loads(embedding_json), dtype=np.float32)
    
    def get_similarity(
        self, 
        text1: str, 
        text2: str,
        normalize: bool = True
    ) -> float:
        """Calculate the cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            normalize: Whether to normalize the embeddings
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        emb1 = self.encode(text1, normalize_embeddings=normalize)
        emb2 = self.encode(text2, normalize_embeddings=normalize)
        return float(np.dot(emb1, emb2))
    
    def get_most_similar(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Find the most similar texts to the query.
        
        Args:
            query: Query text
            texts: List of texts to compare against
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries with 'text', 'index', and 'score' keys
        """
        if not texts:
            return []
        
        # Encode the query and texts
        query_embedding = self.encode(query, normalize_embeddings=True)
        text_embeddings = self.encode(texts, normalize_embeddings=True)
        
        # Calculate cosine similarities
        similarities = np.dot(text_embeddings, query_embedding)
        
        # Get top-k results
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        results = []
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({
                    'text': texts[idx],
                    'index': int(idx),
                    'score': score
                })
        
        # Sort by score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def __call__(self, text: str) -> np.ndarray:
        """Alias for encode() for simpler usage."""
        return self.encode(text)
