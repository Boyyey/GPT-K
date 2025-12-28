"""
FAISS (Facebook AI Similarity Search) index for efficient similarity search.
"""
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger

class FAISSIndex:
    """
    Wrapper around FAISS for efficient similarity search of embeddings.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_dim: int = 384,  # Default for all-MiniLM-L6-v2
        use_gpu: bool = False,
        nprobe: int = 10,
    ):
        """Initialize the FAISS index.
        
        Args:
            index_path: Path to save/load the FAISS index
            embedding_dim: Dimension of the embeddings
            use_gpu: Whether to use GPU for FAISS (if available)
            nprobe: Number of clusters to probe during search (higher = more accurate but slower)
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.nprobe = nprobe
        
        # Initialize empty index
        self.index = None
        self.id_to_idx = {}  # Map memory ID to FAISS index
        self.idx_to_id = {}  # Map FAISS index to memory ID
        self.next_idx = 0
        
        # Initialize FAISS
        self._init_index()
        
        # Load existing index if path is provided
        if self.index_path and os.path.exists(self.index_path):
            self.load(self.index_path)
    
    def _init_index(self) -> None:
        """Initialize a new FAISS index."""
        # Use IndexFlatIP (inner product) for cosine similarity with normalized vectors
        # This is equivalent to cosine similarity when vectors are L2 normalized
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Use GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS index")
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        # Wrap in IndexIDMap to support ID-based operations
        self.index = faiss.IndexIDMap(self.index)
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        memory_ids: List[int],
    ) -> None:
        """Add embeddings to the index.
        
        Args:
            embeddings: Numpy array of shape (n, embedding_dim)
            memory_ids: List of memory IDs corresponding to each embedding
        """
        if len(embeddings) != len(memory_ids):
            raise ValueError("Number of embeddings must match number of memory IDs")
        
        if self.index is None:
            self._init_index()
        
        # Convert to float32 if needed
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Get the next available indices
        indices = np.arange(self.next_idx, self.next_idx + len(embeddings))
        
        # Update ID mappings
        for i, memory_id in enumerate(memory_ids):
            self.id_to_idx[memory_id] = indices[i]
            self.idx_to_id[indices[i]] = memory_id
        
        # Add to index
        self.index.add_with_ids(embeddings, indices)
        self.next_idx += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            k: Number of results to return
            min_score: Minimum similarity score (0-1) for results
            
        Returns:
            List of dicts with 'memory_id' and 'score' for each result
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Convert to float32 if needed and reshape to (1, embedding_dim)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Set nprobe for search
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(self.nprobe, self.index.ntotal)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert to list of results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx < 0:  # Skip invalid indices
                continue
                
            memory_id = self.idx_to_id.get(int(idx))
            if memory_id is not None and score >= min_score:
                results.append({
                    'memory_id': memory_id,
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def remove(self, memory_ids: List[int]) -> None:
        """Remove embeddings from the index by memory IDs."""
        if not memory_ids:
            return
            
        # Get indices to remove
        indices_to_remove = []
        for memory_id in memory_ids:
            if memory_id in self.id_to_idx:
                idx = self.id_to_idx[memory_id]
                indices_to_remove.append(idx)
                
                # Remove from mappings
                del self.id_to_idx[memory_id]
                if idx in self.idx_to_id:
                    del self.idx_to_id[idx]
        
        if indices_to_remove:
            # FAISS requires indices to be int64
            indices_to_remove = np.array(indices_to_remove, dtype=np.int64)
            self.index.remove_ids(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} embeddings from FAISS index")
    
    def clear(self) -> None:
        """Clear all embeddings from the index."""
        self._init_index()  # Reinitialize to clear
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        logger.info("Cleared FAISS index")
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the index and mappings to disk."""
        if not path and not self.index_path:
            raise ValueError("No path provided to save FAISS index")
        
        save_path = path or self.index_path
        base_path = os.path.splitext(save_path)[0]
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{base_path}.faiss")
            
            # Save ID mappings
            with open(f"{base_path}.json", 'w') as f:
                json.dump({
                    'id_to_idx': self.id_to_idx,
                    'next_idx': self.next_idx,
                    'embedding_dim': self.embedding_dim,
                }, f)
                
            logger.info(f"Saved FAISS index to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def load(self, path: Optional[str] = None) -> None:
        """Load the index and mappings from disk."""
        if not path and not self.index_path:
            raise ValueError("No path provided to load FAISS index")
        
        load_path = path or self.index_path
        base_path = os.path.splitext(load_path)[0]
        
        try:
            # Load FAISS index
            if os.path.exists(f"{base_path}.faiss"):
                self.index = faiss.read_index(f"{base_path}.faiss")
                
                # Load ID mappings
                if os.path.exists(f"{base_path}.json"):
                    with open(f"{base_path}.json", 'r') as f:
                        data = json.load(f)
                        self.id_to_idx = {int(k): int(v) for k, v in data['id_to_idx'].items()}
                        self.next_idx = data['next_idx']
                        
                        # Update idx_to_id mapping
                        self.idx_to_id = {v: k for k, v in self.id_to_idx.items()}
                        
                        # Verify embedding dimension
                        if 'embedding_dim' in data and data['embedding_dim'] != self.embedding_dim:
                            logger.warning(
                                f"Loaded embedding dimension {data['embedding_dim']} "
                                f"does not match expected {self.embedding_dim}"
                            )
                
                logger.info(f"Loaded FAISS index from {load_path} with {self.index.ntotal} vectors")
            else:
                logger.warning(f"No FAISS index found at {load_path}, starting with empty index")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Reinitialize on failure
            self._init_index()
            raise
    
    def __len__(self) -> int:
        """Get the number of vectors in the index."""
        return self.index.ntotal if self.index is not None else 0
