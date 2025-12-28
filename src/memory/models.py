"""Database models for the memory system."""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from sqlalchemy import (
    Column, Integer, String, Text, Float, 
    DateTime, ForeignKey, JSON, create_engine, event, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

# SQLAlchemy base class
Base = declarative_base()

class MemoryType(str, Enum):
    """Types of memories in the system."""
    EPISODIC = "episodic"    # Specific events and experiences
    SEMANTIC = "semantic"    # General knowledge and facts
    REFLECTION = "reflection"  # Insights and conclusions
    TASK = "task"           # Tasks and todos
    PREFERENCE = "preference"  # User preferences
    RESPONSE = "response"    # AI responses to user queries

class Memory(Base):
    """Base memory model for storing all types of memories."""
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields
    content = Column(Text, nullable=False)
    memory_type = Column(String(32), nullable=False, index=True)
    source = Column(String(128), default="user")  # Source of the memory
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Metadata
    metadata_ = Column("metadata", JSON, default=dict, nullable=False)
    
    # For semantic search
    embedding = Column(JSON, nullable=True)  # Stored as JSON for portability
    
    # Relationships
    memory_associations = relationship(
        "MemoryAssociation",
        foreign_keys="[MemoryAssociation.memory_id]",
        back_populates="memory"
    )
    
    # Back-reference for related memories
    related_memories = relationship(
        "MemoryAssociation",
        primaryjoin=(
            "or_("
            "MemoryAssociation.memory_id == Memory.id, "
            "MemoryAssociation.related_memory_id == Memory.id"
            ")"
        ),
        viewonly=True
    )
    
    def __repr__(self) -> str:
        return f"<Memory(id={self.id}, type={self.memory_type}, content='{self.content[:50]}...'>"
    
    def to_dict(self, include_related: bool = False) -> Dict[str, Any]:
        """Convert memory to dictionary.
        
        Args:
            include_related: Whether to include related memories
            
        Returns:
            Dictionary representation of the memory
        """
        result = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata_ or {},
            "embedding": self.embedding
        }
        
        if include_related and hasattr(self, 'related_memories'):
            result['related_memories'] = [
                {
                    'id': assoc.id,
                    'relationship_type': assoc.relationship_type,
                    'strength': assoc.strength,
                    'metadata': assoc.metadata_ or {}
                }
                for assoc in self.related_memories
            ]
            
        return result

class MemoryAssociation(Base):
    """Represents relationships between memories."""
    __tablename__ = "memory_associations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(Integer, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    related_memory_id = Column(Integer, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(64), nullable=False)  # e.g., "related_to", "contradicts", etc.
    strength = Column(Float, default=1.0)  # Strength of the relationship (0.0-1.0)
    metadata_ = Column("metadata", JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    memory = relationship("Memory", foreign_keys=[memory_id], back_populates="memory_associations")
    related_memory = relationship("Memory", foreign_keys=[related_memory_id])
    
    def __init__(self, **kwargs):
        """Initialize a new memory association with validation."""
        # Ensure memory_id is always the smaller ID to prevent duplicate relationships
        if 'memory_id' in kwargs and 'related_memory_id' in kwargs:
            if kwargs['memory_id'] > kwargs['related_memory_id']:
                # Swap them to ensure consistent ordering
                kwargs['memory_id'], kwargs['related_memory_id'] = \
                    kwargs['related_memory_id'], kwargs['memory_id']
        
        # Ensure strength is within valid range
        if 'strength' in kwargs:
            kwargs['strength'] = max(0.0, min(1.0, float(kwargs['strength'])))
            
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return (
            f"<MemoryAssociation(id={self.id}, "
            f"memory_id={self.memory_id}, "
            f"related_memory_id={self.related_memory_id}, "
            f"type='{self.relationship_type}')>"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert association to dictionary."""
        return {
            'id': self.id,
            'memory_id': self.memory_id,
            'related_memory_id': self.related_memory_id,
            'relationship_type': self.relationship_type,
            'strength': self.strength,
            'metadata': self.metadata_ or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def update_strength(self, new_strength: float) -> None:
        """Update the strength of this association.
        
        Args:
            new_strength: New strength value (will be clamped to 0.0-1.0)
        """
        self.strength = max(0.0, min(1.0, float(new_strength)))
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update the metadata of this association.
        
        Args:
            updates: Dictionary of metadata updates
        """
        if not hasattr(self, 'metadata_') or not self.metadata_:
            self.metadata_ = {}
        self.metadata_.update(updates)

# Create indexes for performance
Index("idx_memory_type", Memory.memory_type)
Index("idx_memory_source", Memory.source)
Index("idx_memory_created", Memory.created_at)
Index("idx_memory_updated", Memory.updated_at)
Index("idx_memory_association", MemoryAssociation.memory_id, MemoryAssociation.related_memory_id)

def get_engine(db_url: str = "sqlite:///:memory:", **kwargs) -> Engine:
    """Get a SQLAlchemy engine with the specified configuration."""
    if db_url.startswith("sqlite"):
        kwargs.update({
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool
        })
    
    return create_engine(db_url, **kwargs)

def create_session_factory(engine: Engine) -> sessionmaker:
    """Create a session factory for the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(engine: Engine) -> None:
    """Initialize the database with all tables."""
    Base.metadata.create_all(bind=engine)

# SQLite specific optimizations
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints and other SQLite optimizations."""
    if isinstance(dbapi_connection, type("")):
        return  # Skip for in-memory SQLite
    
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-2000")  # 2MB cache
    cursor.close()
