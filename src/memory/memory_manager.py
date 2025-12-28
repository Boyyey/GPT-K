"""Memory manager for the AI assistant's memory system.

Handles storage, retrieval, and search of memories using SQLite and FAISS.
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from sqlalchemy.orm import Session
from loguru import logger

from .models import Memory, MemoryType, MemoryAssociation, get_engine, create_session_factory, init_db
from .embedding import EmbeddingModel
from .faiss_index import FAISSIndex

class MemoryManager:
    """Manages the AI assistant's memory system."""
    
    def __init__(
        self,
        db_url: str = "sqlite:///memories.db",
        embedding_model: Optional[EmbeddingModel] = None,
        faiss_index_path: Optional[str] = None,
        embedding_dim: int = 384,  # Default for all-MiniLM-L6-v2
    ):
        """Initialize the memory manager.
        
        Args:
            db_url: SQLAlchemy database URL
            embedding_model: Pre-initialized embedding model (will create one if None)
            faiss_index_path: Path to save/load the FAISS index
            embedding_dim: Dimension of the embeddings (only used if embedding_model is None)
        """
        # Initialize database
        self.engine = get_engine(db_url)
        self.Session = create_session_factory(self.engine)
        init_db(self.engine)
        
        # Initialize embedding model
        self.embedding_model = embedding_model or EmbeddingModel()
        
        # Initialize FAISS index
        self.faiss_index = FAISSIndex(
            index_path=faiss_index_path or "memories.faiss",
            embedding_dim=self.embedding_model.embedding_dim
        )
        
        # Load existing memories into FAISS index
        self._index_existing_memories()
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_model.embedding_dim
    
    def _index_existing_memories(self) -> None:
        """Load existing memories from the database into the FAISS index."""
        with self.Session() as session:
            # Get all memories with embeddings
            memories = session.query(Memory).filter(Memory.embedding.isnot(None)).all()
            
            if not memories:
                logger.info("No existing memories with embeddings found in database")
                return
            
            # Prepare data for FAISS
            memory_ids = []
            embeddings = []
            
            for memory in memories:
                try:
                    embedding = memory.embedding
                    if isinstance(embedding, str):
                        embedding = json.loads(embedding)
                    
                    memory_ids.append(memory.id)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to process memory {memory.id}: {e}")
            
            if memory_ids:
                self.faiss_index.add_embeddings(
                    np.array(embeddings, dtype=np.float32),
                    memory_ids
                )
                logger.info(f"Indexed {len(memory_ids)} existing memories")
    
    def add_memory(
        self,
        content: str,
        memory_type: Union[MemoryType, str],
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True,
        session: Optional[Session] = None,
    ) -> Memory:
        """Add a new memory to the system.
        
        Args:
            content: The content of the memory
            memory_type: Type of memory (episodic, semantic, etc.)
            source: Source of the memory (e.g., 'user', 'system')
            metadata: Additional metadata for the memory
            generate_embedding: Whether to generate an embedding for the memory
            session: Optional existing database session to use
            
        Returns:
            The created Memory object
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type.lower())
        
        memory = Memory(
            content=content,
            memory_type=memory_type.value,
            source=source,
            metadata_=metadata or {},
            embedding=None
        )
        
        # Generate embedding if requested
        if generate_embedding:
            try:
                embedding = self.embedding_model.encode(content)
                memory.embedding = json.dumps(embedding.tolist())
            except Exception as e:
                logger.error(f"Failed to generate embedding for memory: {e}")
        
        # Save to database
        local_session = None
        try:
            if session is None:
                local_session = self.Session()
                session = local_session
            
            session.add(memory)
            session.flush()  # Get the memory ID
            
            # Add to FAISS index if we have an embedding
            if memory.embedding:
                self.faiss_index.add_embeddings(
                    np.array([embedding], dtype=np.float32),
                    [memory.id]
                )
            
            if local_session:
                session.commit()
                
            logger.info(f"Added memory {memory.id} (type: {memory_type})")
            return memory
            
        except Exception as e:
            if local_session:
                session.rollback()
            logger.error(f"Failed to add memory: {e}")
            raise
            
        finally:
            if local_session:
                local_session.close()
    
    def get_memory(self, memory_id: int, session: Optional[Session] = None) -> Optional[Memory]:
        """Get a memory by ID."""
        local_session = None
        try:
            if session is None:
                local_session = self.Session()
                session = local_session
            
            memory = session.query(Memory).get(memory_id)
            return memory
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
            
        finally:
            if local_session:
                local_session.close()
    
    def search_memories(
        self,
        query: str,
        memory_types: Optional[List[Union[MemoryType, str]]] = None,
        limit: int = 5,
        min_score: float = 0.5,
        session: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories similar to the query.
        
        Args:
            query: The search query
            memory_types: Optional list of memory types to filter by
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            session: Optional existing database session to use
            
        Returns:
            List of dictionaries with memory details and similarity scores
        """
        if not query.strip():
            return []
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_model.encode(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Search FAISS index
        faiss_results = self.faiss_index.search(
            query_embedding,
            k=limit * 2,  # Get extra results to account for filtering
            min_score=min_score
        )
        
        if not faiss_results:
            return []
        
        # Convert memory types to strings for filtering
        type_filters = None
        if memory_types:
            type_filters = {t.value if isinstance(t, MemoryType) else t.lower() for t in memory_types}
        
        # Get full memory objects from database
        local_session = None
        try:
            if session is None:
                local_session = self.Session()
                session = local_session
            
            results = []
            memory_ids = [r['memory_id'] for r in faiss_results]
            
            # Get memories in batches to avoid very large IN clauses
            batch_size = 100
            for i in range(0, len(memory_ids), batch_size):
                batch_ids = memory_ids[i:i + batch_size]
                
                query = session.query(Memory).filter(Memory.id.in_(batch_ids))
                
                # Apply memory type filters if specified
                if type_filters:
                    query = query.filter(Memory.memory_type.in_(type_filters))
                
                # Get memories and preserve order from FAISS results
                memories = {m.id: m for m in query.all()}
                
                # Add to results with scores
                for faiss_result in faiss_results:
                    memory_id = faiss_result['memory_id']
                    if memory_id in memories:
                        memory = memories[memory_id]
                        result = {
                            'memory': memory,
                            'score': faiss_result['score'],
                            'rank': faiss_result['rank']
                        }
                        results.append(result)
                        
                        # Stop if we have enough results
                        if len(results) >= limit:
                            break
                
                if len(results) >= limit:
                    break
            
            # Sort by score (descending)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Update last_accessed timestamp for retrieved memories
            memory_ids = [r['memory'].id for r in results]
            if memory_ids:
                session.query(Memory).filter(Memory.id.in_(memory_ids))\
                    .update({
                        Memory.last_accessed: datetime.utcnow()
                    }, synchronize_session=False)
                
                if local_session:
                    session.commit()
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
            
        finally:
            if local_session:
                local_session.close()
    
    def delete_memory(
        self,
        memory_id: int,
        session: Optional[Session] = None,
    ) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            session: Optional existing database session to use
            
        Returns:
            True if the memory was deleted, False otherwise
        """
        local_session = None
        try:
            if session is None:
                local_session = self.Session()
                session = local_session
                
            memory = session.query(Memory).get(memory_id)
            if not memory:
                return False
                
            # Remove from FAISS index
            self.faiss_index.remove([memory_id])
            
            # Delete from database
            session.delete(memory)
            
            if local_session:
                session.commit()
                
            logger.info(f"Deleted memory {memory_id}")
            return True
            
        except Exception as e:
            if local_session:
                session.rollback()
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
            
        finally:
            if local_session:
                local_session.close()
    
    def add_memory_association(
        self,
        memory_id: int,
        related_memory_id: int,
        relationship_type: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> Optional[MemoryAssociation]:
        """Create an association between two memories.
        
        Args:
            memory_id: ID of the first memory
            related_memory_id: ID of the second memory
            relationship_type: Type of relationship (e.g., 'related_to', 'contradicts')
            strength: Strength of the relationship (0.0-1.0)
            metadata: Additional metadata for the relationship
            session: Optional existing database session to use
            
        Returns:
            The created MemoryAssociation object, or None if failed
        """
        # Ensure memory_id is the smaller ID to avoid duplicate relationships
        if memory_id > related_memory_id:
            memory_id, related_memory_id = related_memory_id, memory_id
        
        local_session = None
        try:
            if session is None:
                local_session = self.Session()
                session = local_session
            
            # Check if association already exists
            existing = session.query(MemoryAssociation).filter(
                MemoryAssociation.memory_id == memory_id,
                MemoryAssociation.related_memory_id == related_memory_id,
                MemoryAssociation.relationship_type == relationship_type
            ).first()
            
            if existing:
                # Update existing association
                existing.strength = strength
                existing.metadata_ = metadata or {}
            else:
                # Create new association
                association = MemoryAssociation(
                    memory_id=memory_id,
                    related_memory_id=related_memory_id,
                    relationship_type=relationship_type,
                    strength=strength,
                    metadata_=metadata or {}
                )
                session.add(association)
                
            if local_session:
                session.commit()
                
            return existing or association
            
        except Exception as e:
            if local_session:
                session.rollback()
            logger.error(
                f"Failed to create memory association between {memory_id} and "
                f"{related_memory_id}: {e}"
            )
            return None
            
        finally:
            if local_session:
                local_session.close()
    
    def get_related_memories(
        self,
        memory_id: int,
        relationship_type: Optional[str] = None,
        min_strength: float = 0.0,
        session: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Get memories related to the specified memory.
        
        Args:
            memory_id: ID of the memory to find relations for
            relationship_type: Optional filter for relationship type
            min_strength: Minimum strength of relationships to include
            session: Optional existing database session to use
            
        Returns:
            List of dictionaries with 'memory' and 'relationship' keys
        """
        local_session = None
        try:
            if session is None:
                local_session = self.Session()
                session = local_session
            
            query = session.query(
                Memory,
                MemoryAssociation.relationship_type,
                MemoryAssociation.strength
            ).join(
                MemoryAssociation,
                (Memory.id == MemoryAssociation.memory_id) | 
                (Memory.id == MemoryAssociation.related_memory_id)
            ).filter(
                (MemoryAssociation.memory_id == memory_id) | 
                (MemoryAssociation.related_memory_id == memory_id),
                Memory.id != memory_id,
                MemoryAssociation.strength >= min_strength
            )
            
            if relationship_type:
                query = query.filter(MemoryAssociation.relationship_type == relationship_type)
            
            results = []
            for memory, rel_type, strength in query.all():
                results.append({
                    'memory': memory,
                    'relationship': {
                        'type': rel_type,
                        'strength': strength
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get related memories for {memory_id}: {e}")
            return []
            
        finally:
            if local_session:
                local_session.close()
    
    def cleanup_old_memories(
        self,
        max_age_days: int = 30,
        min_importance: float = 0.3,
        memory_types: Optional[List[Union[MemoryType, str]]] = None,
        dry_run: bool = True,
        session: Optional[Session] = None,
    ) -> Dict[str, int]:
        """Clean up old or low-importance memories.
        
        Args:
            max_age_days: Maximum age of memories to keep (in days)
            min_importance: Minimum importance score to keep (0.0-1.0)
            memory_types: Optional list of memory types to clean up
            dry_run: If True, only return counts without deleting
            session: Optional existing database session to use
            
        Returns:
            Dictionary with cleanup statistics
        """
        local_session = None
        try:
            if session is None:
                local_session = self.Session()
                session = local_session
            
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            # Build the base query
            query = session.query(Memory).filter(
                Memory.last_accessed < cutoff_date,
                Memory.metadata_['importance'].as_float() < min_importance
            )
            
            # Apply memory type filters if specified
            if memory_types:
                type_filters = [t.value if isinstance(t, MemoryType) else t.lower() 
                              for t in memory_types]
                query = query.filter(Memory.memory_type.in_(type_filters))
            
            # Get memory IDs to delete
            memory_ids = [m.id for m in query.all()]
            
            stats = {
                'total_found': len(memory_ids),
                'deleted': 0,
                'skipped': 0
            }
            
            if not dry_run and memory_ids:
                # Delete from FAISS index
                self.faiss_index.remove(memory_ids)
                
                # Delete from database
                deleted_count = session.query(Memory).filter(
                    Memory.id.in_(memory_ids)
                ).delete(synchronize_session=False)
                
                if local_session:
                    session.commit()
                    
                stats['deleted'] = deleted_count
                stats['skipped'] = len(memory_ids) - deleted_count
                
                logger.info(f"Cleaned up {deleted_count} old/low-importance memories")
            
            return stats
            
        except Exception as e:
            if local_session:
                session.rollback()
            logger.error(f"Failed to clean up memories: {e}")
            return {'error': str(e)}
            
        finally:
            if local_session:
                local_session.close()
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save the FAISS index to disk."""
        self.faiss_index.save(path)
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            self.save_index()
            if hasattr(self, 'engine'):
                self.engine.dispose()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
