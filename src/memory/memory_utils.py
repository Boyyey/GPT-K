"""Utility functions for working with the memory system."""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from .embedding import EmbeddingModel
from .models import Memory, MemoryAssociation, MemoryType

# Regular expression for extracting entities (simple version)
ENTITY_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

def extract_entities(text: str) -> List[str]:
    """Extract potential named entities from text.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of potential named entities
    """
    return ENTITY_PATTERN.findall(text)

def calculate_memory_importance(
    content: str,
    memory_type: MemoryType,
    source: str = "system",
    existing_memories: Optional[List[Memory]] = None,
    embedding_model: Optional[EmbeddingModel] = None
) -> float:
    """Calculate the importance score for a new memory.
    
    Args:
        content: The content of the memory
        memory_type: Type of the memory
        source: Source of the memory
        existing_memories: Optional list of existing memories for context
        embedding_model: Optional embedding model for semantic analysis
        
    Returns:
        Importance score between 0.0 and 1.0
    """
    # Base importance based on memory type
    type_weights = {
        MemoryType.PREFERENCE: 0.9,
        MemoryType.REFLECTION: 0.8,
        MemoryType.EPISODIC: 0.7,
        MemoryType.SEMANTIC: 0.6,
        MemoryType.TASK: 0.5
    }
    
    importance = type_weights.get(memory_type, 0.5)
    
    # Adjust based on content length (longer content might be more important)
    content_length = len(content.split())
    length_factor = min(1.0, content_length / 50)  # Cap at 1.0 for 50+ words
    importance = (importance * 0.7) + (length_factor * 0.3)
    
    # Adjust based on entities (more entities might indicate more important information)
    entities = extract_entities(content)
    if entities:
        entity_factor = min(1.0, len(entities) / 5)  # Cap at 1.0 for 5+ entities
        importance = (importance * 0.8) + (entity_factor * 0.2)
    
    # If we have an embedding model and existing memories, check for novelty
    if embedding_model and existing_memories:
        try:
            # Get embeddings for all memories
            memory_texts = [m.content for m in existing_memories] + [content]
            embeddings = embedding_model.encode(memory_texts, normalize_embeddings=True)
            
            # Calculate average similarity to existing memories
            new_embedding = embeddings[-1:]
            existing_embeddings = embeddings[:-1]
            
            if len(existing_embeddings) > 0:
                similarities = np.dot(existing_embeddings, new_embedding.T).flatten()
                avg_similarity = float(np.mean(similarities))
                
                # Novel information gets higher importance
                novelty_factor = 1.0 - avg_similarity
                importance = (importance * 0.7) + (novelty_factor * 0.3)
                
        except Exception as e:
            logger.warning(f"Error calculating memory novelty: {e}")
    
    # Ensure the score is within bounds
    return max(0.0, min(1.0, importance))

def find_related_memories(
    memory: Memory,
    session: Session,
    threshold: float = 0.7,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Find memories related to the given memory.
    
    Args:
        memory: The memory to find related memories for
        session: Database session
        threshold: Minimum similarity threshold (0.0-1.0)
        limit: Maximum number of related memories to return
        
    Returns:
        List of dictionaries with 'memory' and 'score' keys
    """
    if not memory.embedding:
        return []
    
    try:
        # Get all memories with embeddings (excluding the current one)
        memories = session.query(Memory).filter(
            Memory.id != memory.id,
            Memory.embedding.isnot(None)
        ).all()
        
        if not memories:
            return []
        
        # Convert embeddings to numpy arrays
        embeddings = []
        valid_memories = []
        
        for m in memories:
            try:
                if isinstance(m.embedding, str):
                    emb = json.loads(m.embedding)
                else:
                    emb = m.embedding
                
                if emb and len(emb) > 0:
                    embeddings.append(emb)
                    valid_memories.append(m)
            except Exception as e:
                logger.warning(f"Error processing memory {m.id}: {e}")
        
        if not embeddings:
            return []
        
        # Calculate similarities
        memory_embedding = json.loads(memory.embedding) if isinstance(memory.embedding, str) else memory.embedding
        similarities = np.dot(embeddings, memory_embedding)
        
        # Get top matches
        top_indices = np.argpartition(similarities, -limit)[-limit:]
        
        # Create results
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append({
                    'memory': valid_memories[idx],
                    'score': float(similarities[idx])
                })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
        
    except Exception as e:
        logger.error(f"Error finding related memories: {e}")
        return []

def merge_similar_memories(
    memory1: Memory,
    memory2: Memory,
    session: Session,
    similarity_threshold: float = 0.9
) -> Optional[Memory]:
    """Merge two similar memories into one.
    
    Args:
        memory1: First memory to merge
        memory2: Second memory to merge
        session: Database session
        similarity_threshold: Minimum similarity required for merging
        
    Returns:
        The merged memory, or None if memories were not similar enough
    """
    # Check if memories are already the same
    if memory1.id == memory2.id:
        return memory1
    
    # Check if memories are similar enough
    if not memory1.embedding or not memory2.embedding:
        return None
    
    try:
        # Calculate similarity
        emb1 = json.loads(memory1.embedding) if isinstance(memory1.embedding, str) else memory1.embedding
        emb2 = json.loads(memory2.embedding) if isinstance(memory2.embedding, str) else memory2.embedding
        
        similarity = np.dot(emb1, emb2)
        
        if similarity < similarity_threshold:
            return None
            
        # Merge the memories with proper newlines
        merged_content = '\n\n'.join([
            memory1.content.strip(),
            memory2.content.strip()
        ])
        
        # Keep the more recent updated_at timestamp
        updated_at = max(
            memory1.updated_at or datetime.min,
            memory2.updated_at or datetime.min
        )
        
        # Update memory1 with merged content
        memory1.content = merged_content
        memory1.updated_at = updated_at
        
        # Update metadata
        if memory2.metadata_:
            if not memory1.metadata_:
                memory1.metadata_ = {}
            memory1.metadata_.update(memory2.metadata_)
        
        # Transfer associations
        for assoc in memory2.related_memories:
            if assoc.memory_id == memory2.id:
                other_id = assoc.related_memory_id
            else:
                other_id = assoc.memory_id
                
            if other_id != memory1.id:  # Don't create self-references
                # Check if association already exists
                existing = session.query(MemoryAssociation).filter(
                    ((MemoryAssociation.memory_id == memory1.id) & 
                     (MemoryAssociation.related_memory_id == other_id)) |
                    ((MemoryAssociation.memory_id == other_id) & 
                     (MemoryAssociation.related_memory_id == memory1.id))
                ).first()
                
                if not existing:
                    # Create new association
                    new_assoc = MemoryAssociation(
                        memory_id=min(memory1.id, other_id),
                        related_memory_id=max(memory1.id, other_id),
                        relationship_type=assoc.relationship_type,
                        strength=assoc.strength,
                        metadata_=assoc.metadata_
                    )
                    session.add(new_assoc)
        
        # Delete the second memory
        session.delete(memory2)
        session.commit()
        
        return memory1
        
    except Exception as e:  # pylint: disable=broad-except
        session.rollback()
        logger.error("Error merging memories %s and %s: %s", 
                    memory1.id, memory2.id, str(e))
        return None

def get_memory_statistics(session: Session) -> Dict[str, Any]:
    """Get statistics about the memory database.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    # Basic counts
    stats['total_memories'] = session.query(Memory).count()
    
    # Count by type
    type_counts = {}
    for mem_type in MemoryType:
        count = session.query(Memory).filter(
            Memory.memory_type == mem_type.value
        ).count()
        type_counts[mem_type.value] = count
    stats['count_by_type'] = type_counts
    
    # Count of memories with/without embeddings
    stats['with_embedding'] = session.query(Memory).filter(
        Memory.embedding.isnot(None)
    ).count()
    
    stats['without_embedding'] = stats['total_memories'] - stats['with_embedding']
    
    # Memory age statistics
    oldest = session.query(Memory).order_by(Memory.created_at).first()
    newest = session.query(Memory).order_by(Memory.created_at.desc()).first()
    
    if oldest and newest:
        stats['oldest_memory'] = oldest.created_at.isoformat()
        stats['newest_memory'] = newest.created_at.isoformat()
        
        if oldest != newest:
            age_range = newest.created_at - oldest.created_at
            stats['age_range_days'] = age_range.days
    
    # Association statistics
    stats['total_associations'] = session.query(MemoryAssociation).count()
    
    # Average associations per memory
    if stats['total_memories'] > 0:
        stats['avg_associations_per_memory'] = (
            stats['total_associations'] / stats['total_memories']
        )
    
    return stats
