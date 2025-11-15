"""
Test for retriever to find top 2 documents from library by query.
"""

import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.vectore_store import VectorStore
from src.rag.retriver import Retriever


class TestRetriever(unittest.TestCase):
    """Test cases for the retriever."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Initialize vector store and add documents (using in-memory mode - no Docker needed)
        cls.vector_store = VectorStore(
            collection_name="test_construction_materials",
            use_in_memory=True  # No Docker required for tests
        )
        
        # Load and add documents
        data_path = "data/raw/raw_materials.jsonl"
        cls.vector_store.add_documents(data_path)
        
        # Initialize retriever with shared client (important for in-memory mode!)
        cls.retriever = Retriever(
            collection_name="test_construction_materials",
            qdrant_client=cls.vector_store.qdrant_client  # Share the same client!
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Delete test collection
        try:
            cls.vector_store.delete_collection()
        except Exception:
            pass
    
    def test_retrieve_top_2(self):
        """Test retrieving top 2 documents by query."""
        query = "бетон М400 характеристики"
        
        # Retrieve top 2
        results = self.retriever.retrieve(query, top_k=2)
        
        # Assertions
        self.assertEqual(len(results), 2, "Should return exactly 2 results")
        
        # Check that results have required fields
        for result in results:
            self.assertIn('id', result, "Result should have 'id' field")
            self.assertIn('score', result, "Result should have 'score' field")
            self.assertIn('text', result, "Result should have 'text' field")
            self.assertIn('url', result, "Result should have 'url' field")
            
            # Score should be between 0 and 1 for cosine similarity
            self.assertGreaterEqual(result['score'], 0, "Score should be >= 0")
            self.assertLessEqual(result['score'], 1, "Score should be <= 1")
        
        # Results should be sorted by score (descending)
        scores = [r['score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True), 
                        "Results should be sorted by score descending")
    
    def test_retrieve_different_query(self):
        """Test retrieving top 2 documents with a different query."""
        query = "щебень гранитный фракция"
        
        results = self.retriever.retrieve(query, top_k=2)
        
        self.assertEqual(len(results), 2, "Should return exactly 2 results")
        
        # Verify all results have text content
        for result in results:
            self.assertIsInstance(result['text'], str, "Text should be a string")
            self.assertGreater(len(result['text']), 0, "Text should not be empty")
    
    def test_retrieve_empty_query(self):
        """Test that empty query still returns results."""
        query = ""
        
        results = self.retriever.retrieve(query, top_k=2)
        
        # Should still return results (though they may not be relevant)
        self.assertIsInstance(results, list, "Should return a list")
    
    def test_retrieve_top_k_method(self):
        """Test the retrieve_top_k alias method."""
        query = "песок карьерный"
        
        results = self.retriever.retrieve_top_k(query, k=2)
        
        self.assertEqual(len(results), 2, "Should return exactly 2 results")
        
        # Verify structure
        for result in results:
            self.assertIn('score', result)
            self.assertIn('text', result)
            self.assertIn('url', result)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

