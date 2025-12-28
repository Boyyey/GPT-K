"""
Tests for the memory system.
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.memory_manager import MemoryManager
from src.memory.models import Memory, MemoryType, MemoryAssociation, Base
from src.memory.embedding import EmbeddingModel

class TestMemorySystem(unittest.TestCase):
    """Test cases for the memory system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary database
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db_url = f"sqlite:///{self.db_path}"
        
        # Create a temporary FAISS index
        self.faiss_path = f"{self.db_path}.faiss"
        
        # Initialize the memory manager
        self.memory_manager = MemoryManager(
            db_url=self.db_url,
            faiss_index_path=self.faiss_path
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Close the memory manager
        self.memory_manager.close()
        
        # Remove temporary files
        try:
            os.close(self.db_fd)
            os.unlink(self.db_path)
            if os.path.exists(self.faiss_path):
                os.unlink(self.faiss_path)
        except Exception as e:
            print(f"Warning: Failed to clean up test files: {e}")
    
    def test_add_memory(self):
        """Test adding a memory."""
        # Add a memory
        memory = self.memory_manager.add_memory(
            content="The user's favorite color is blue",
            memory_type=MemoryType.SEMANTIC,
            source="test",
            metadata={"importance": 0.8}
        )
        
        # Check that the memory was added
        self.assertIsNotNone(memory.id)
        self.assertEqual(memory.content, "The user's favorite color is blue")
        self.assertEqual(memory.memory_type, MemoryType.SEMANTIC.value)
        self.assertEqual(memory.source, "test")
        self.assertEqual(memory.metadata_["importance"], 0.8)
        self.assertIsNotNone(memory.embedding)
    
    def test_get_memory(self):
        """Test retrieving a memory by ID."""
        # Add a memory
        memory = self.memory_manager.add_memory(
            content="Test memory",
            memory_type=MemoryType.EPISODIC,
            source="test"
        )
        
        # Retrieve the memory
        retrieved = self.memory_manager.get_memory(memory.id)
        
        # Check that the memory was retrieved correctly
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, memory.id)
        self.assertEqual(retrieved.content, "Test memory")
    
    def test_search_memories(self):
        """Test searching memories by semantic similarity."""
        # Add some test memories
        memories = [
            ("The sky is blue", MemoryType.SEMANTIC, "system"),
            ("I like pizza", MemoryType.EPISODIC, "user"),
            ("Python is a programming language", MemoryType.SEMANTIC, "system"),
        ]
        
        for content, mem_type, source in memories:
            self.memory_manager.add_memory(
                content=content,
                memory_type=mem_type,
                source=source
            )
        
        # Search for a related concept
        results = self.memory_manager.search_memories(
            "What color is the sky?",
            memory_types=[MemoryType.SEMANTIC],
            limit=1
        )
        
        # Check that we got a result
        self.assertEqual(len(results), 1)
        self.assertIn("sky", results[0]['memory'].content.lower())
        self.assertGreater(results[0]['score'], 0.5)  # Should be quite relevant
    
    def test_memory_associations(self):
        """Test creating and retrieving memory associations."""
        # Add two memories
        mem1 = self.memory_manager.add_memory(
            content="I like cats",
            memory_type=MemoryType.EPISODIC,
            source="user"
        )
        
        mem2 = self.memory_manager.add_memory(
            content="Cats are furry animals",
            memory_type=MemoryType.SEMANTIC,
            source="system"
        )
        
        # Create an association
        self.memory_manager.add_memory_association(
            memory_id=mem1.id,
            related_memory_id=mem2.id,
            relationship_type="related_to",
            strength=0.9
        )
        
        # Get related memories
        related = self.memory_manager.get_related_memories(mem1.id)
        
        # Check the association
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]['memory'].id, mem2.id)
        self.assertEqual(related[0]['relationship']['type'], "related_to")
        self.assertAlmostEqual(related[0]['relationship']['strength'], 0.9)
    
    def test_cleanup_old_memories(self):
        """Test cleaning up old memories."""
        # Add a memory with old last_accessed time
        old_time = datetime.utcnow() - timedelta(days=60)  # 60 days old
        
        with self.memory_manager.Session() as session:
            memory = Memory(
                content="Old memory",
                memory_type=MemoryType.EPISODIC.value,
                source="test",
                last_accessed=old_time,
                metadata_={"importance": 0.2}  # Low importance
            )
            session.add(memory)
            session.commit()
            memory_id = memory.id
        
        # Add a recent memory
        recent_memory = self.memory_manager.add_memory(
            content="Recent memory",
            memory_type=MemoryType.EPISODIC,
            source="test",
            metadata={"importance": 0.8}  # High importance
        )
        
        # Run cleanup (should remove the old, low-importance memory)
        stats = self.memory_manager.cleanup_old_memories(
            max_age_days=30,
            min_importance=0.3,
            dry_run=False
        )
        
        # Check the stats
        self.assertEqual(stats['total_found'], 1)
        self.assertEqual(stats['deleted'], 1)
        
        # Check that the old memory was deleted
        self.assertIsNone(self.memory_manager.get_memory(memory_id))
        
        # Check that the recent memory still exists
        self.assertIsNotNone(self.memory_manager.get_memory(recent_memory.id))
    
    def test_persistence(self):
        """Test that memories persist between sessions."""
        # Add a memory
        memory = self.memory_manager.add_memory(
            content="Persistent memory",
            memory_type=MemoryType.SEMANTIC,
            source="test"
        )
        memory_id = memory.id
        
        # Close the current manager
        self.memory_manager.close()
        
        # Create a new manager with the same paths
        new_manager = MemoryManager(
            db_url=self.db_url,
            faiss_index_path=self.faiss_path
        )
        
        try:
            # Check that the memory was loaded
            loaded = new_manager.get_memory(memory_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.content, "Persistent memory")
            
            # Test that search works with the loaded index
            results = new_manager.search_memories("persistent")
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]['memory'].id, memory_id)
            
        finally:
            new_manager.close()

if __name__ == "__main__":
    unittest.main()
