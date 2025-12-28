# Memory System

This module implements a persistent memory system for the AI assistant, allowing it to store, retrieve, and search through memories efficiently.

## Features

- **Persistent Storage**: Memories are stored in an SQLite database for persistence across sessions
- **Semantic Search**: Find relevant memories using semantic similarity
- **Memory Types**: Support for different types of memories (episodic, semantic, reflection, etc.)
- **Associations**: Create relationships between different memories
- **Efficient Search**: FAISS-based indexing for fast similarity search
- **Embedding Support**: Built-in support for sentence-transformers embeddings

## Components

### 1. Memory Models

- `Memory`: Base model for storing memory content, type, metadata, and embeddings
- `MemoryAssociation`: Represents relationships between memories
- `MemoryType`: Enum for different types of memories

### 2. Embedding Model

- `EmbeddingModel`: Handles text embedding generation using sentence-transformers
  - Supports various pre-trained models
  - Handles batch processing
  - Includes similarity calculation utilities

### 3. FAISS Index

- `FAISSIndex`: Wrapper around FAISS for efficient similarity search
  - Fast nearest neighbor search
  - Support for GPU acceleration
  - Persistent storage of the index

### 4. Memory Manager

- `MemoryManager`: High-level interface for working with memories
  - Add, retrieve, update, and delete memories
  - Search memories by semantic similarity
  - Manage memory associations
  - Clean up old or low-importance memories

## Usage

### Basic Usage

```python
from memory.memory_manager import MemoryManager
from memory.models import MemoryType

# Initialize the memory manager
with MemoryManager("sqlite:///memories.db") as memory_manager:
    # Add a memory
    memory = memory_manager.add_memory(
        content="The user's favorite color is blue",
        memory_type=MemoryType.SEMANTIC,
        source="user_input",
        metadata={"importance": 0.8}
    )
    
    # Search for similar memories
    results = memory_manager.search_memories(
        "What color does the user like?",
        memory_types=[MemoryType.SEMANTIC],
        limit=3
    )
    
    for result in results:
        print(f"Score: {result['score']:.3f} - {result['memory'].content}")
```

### Advanced Usage

```python
from memory.memory_manager import MemoryManager
from memory.models import MemoryType

# Initialize with custom settings
memory_manager = MemoryManager(
    db_url="sqlite:///memories.db",
    faiss_index_path="memories.faiss",
)

try:
    # Add multiple memories
    memories = [
        ("The sky is blue", MemoryType.SEMANTIC, "system"),
        ("I like pizza", MemoryType.EPISODIC, "user"),
        ("Python is a programming language", MemoryType.SEMANTIC, "system"),
    ]
    
    for content, mem_type, source in memories:
        memory_manager.add_memory(
            content=content,
            memory_type=mem_type,
            source=source,
            metadata={"importance": 0.7}
        )
    
    # Create associations between memories
    memory_manager.add_memory_association(
        memory_id=1,
        related_memory_id=3,
        relationship_type="related_to",
        strength=0.8
    )
    
    # Get related memories
    related = memory_manager.get_related_memories(
        memory_id=1,
        relationship_type="related_to"
    )
    
    # Clean up old memories
    stats = memory_manager.cleanup_old_memories(
        max_age_days=30,
        min_importance=0.3,
        dry_run=False
    )
    
finally:
    # Ensure resources are properly cleaned up
    memory_manager.close()
```

## Configuration

The memory system can be configured using the following environment variables:

- `MEMORY_DB_URL`: Database connection URL (default: `sqlite:///memories.db`)
- `FAISS_INDEX_PATH`: Path to the FAISS index file (default: `memories.faiss`)
- `EMBEDDING_MODEL`: Name of the sentence-transformers model to use (default: `all-MiniLM-L6-v2`)
- `USE_GPU`: Whether to use GPU for embeddings and FAISS (default: `False`)

## Dependencies

- SQLAlchemy >= 2.0.0
- sentence-transformers >= 2.2.2
- faiss-cpu >= 1.7.0 (or faiss-gpu for GPU support)
- numpy >= 1.21.0
- python-dateutil >= 2.8.2

## Notes

- The system automatically creates the necessary database tables on first run
- Embeddings are generated on-the-fly when adding new memories
- The FAISS index is automatically saved when the manager is closed
- For production use, consider using a more robust database like PostgreSQL

## License

This project is licensed under the MIT License - see the LICENSE file for details.
