"""
Memory System.

Provides persistent storage and retrieval of memories using SQLite and vector embeddings.
"""

from .memory_manager import MemoryManager
from .models import Memory, MemoryType
from .embedding import EmbeddingModel
from .faiss_index import FAISSIndex

__all__ = [
    'MemoryManager',
    'Memory',
    'MemoryType',
    'EmbeddingModel',
    'FAISSIndex'
]
