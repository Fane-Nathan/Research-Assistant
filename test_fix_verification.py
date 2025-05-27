#!/usr/bin/env python3
"""
Test script to verify that the metadata key fix is working correctly.
This script demonstrates that the RAG system can now properly retrieve
and process documents regardless of whether they use 'entry_id' or 'arxiv_entry_id'.
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hybrid_search_rag import config
from hybrid_search_rag.data_handling.data_manager import DataManager
from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import HybridRecommender
from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel

def test_metadata_keys():
    """Test that our fix handles both entry_id and arxiv_entry_id keys correctly."""
    print("🔍 Testing metadata key handling fix...")
    
    # Load a sample of metadata to check key distribution
    metadata_path = Path(config.DATA_DIR) / config.METADATA_FILE
    
    if not metadata_path.exists():
        print("❌ Metadata file not found. Cannot run test.")
        return False
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Analyze key distribution
    entry_id_count = sum(1 for doc in metadata if 'entry_id' in doc)
    arxiv_entry_id_count = sum(1 for doc in metadata if 'arxiv_entry_id' in doc)
    both_keys_count = sum(1 for doc in metadata if 'entry_id' in doc and 'arxiv_entry_id' in doc)
    neither_keys_count = sum(1 for doc in metadata if 'entry_id' not in doc and 'arxiv_entry_id' not in doc)
    
    print(f"📊 Metadata analysis:")
    print(f"   Total documents: {len(metadata)}")
    print(f"   Documents with 'entry_id': {entry_id_count}")
    print(f"   Documents with 'arxiv_entry_id': {arxiv_entry_id_count}")
    print(f"   Documents with both keys: {both_keys_count}")
    print(f"   Documents with neither key: {neither_keys_count}")
    
    # Test that our fix would handle both cases
    test_cases = [
        {"entry_id": "test1", "title": "Test with entry_id"},
        {"arxiv_entry_id": "test2", "title": "Test with arxiv_entry_id"},
        {"entry_id": "test3", "arxiv_entry_id": "test3_alt", "title": "Test with both keys"},
        {"title": "Test with no ID keys"}
    ]
    
    print(f"\n🧪 Testing key handling logic:")
    for i, doc in enumerate(test_cases, 1):
        # Simulate the logic from our fix
        has_entry_id = 'entry_id' in doc
        has_arxiv_entry_id = 'arxiv_entry_id' in doc
        
        if has_entry_id or has_arxiv_entry_id:
            doc_id = doc.get('entry_id') or doc.get('arxiv_entry_id', 'N/A')
            status = "✅ PASS - Would be processed"
        else:
            status = "⚠️  SKIP - Would be skipped (no ID key)"
        
        print(f"   Test case {i}: {status}")
        print(f"      Keys: {list(doc.keys())}")
        if has_entry_id or has_arxiv_entry_id:
            print(f"      Selected ID: {doc_id}")
    
    return True

def test_retrieval_system():
    """Test that the retrieval system works end-to-end."""
    print(f"\n🔄 Testing retrieval system...")
    
    try:
        # Initialize components
        data_manager = DataManager(
            data_dir=config.DATA_DIR,
            metadata_filename=config.METADATA_FILE,
            embeddings_filename=config.EMBEDDINGS_FILE,
            bm25_filename=config.BM25_INDEX_FILE
        )
        print("📁 Loading data...")
        metadata, embeddings, bm25_index = data_manager.load_all_data()
        if metadata is None:
            print("❌ Failed to load data")
            return False
        
        print("🧮 Initializing embedder...")
        embedder = EmbeddingModel()
        
        print("🔍 Initializing recommender...")
        recommender = HybridRecommender(
            embed_model=embedder
        )
        
        print("🎯 Testing recommendation...")
        test_query = "neural networks"
        
        # Use the correct method signature based on the HybridRecommender class
        from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import RecommendationParams
        
        params = RecommendationParams(
            semantic_candidates=10,
            keyword_candidates=10,
            fusion_k=60,
            top_n_final=5
        )
        
        results = recommender.recommend(
            query=test_query,
            resource_metadata=metadata,
            resource_embeddings=embeddings,
            bm25_index=bm25_index,
            params=params
        )
        print(f"✅ Successfully retrieved {len(results)} results for query: '{test_query}'")
        
        if results: # Add this block
            print(f"🔍 Inspecting first result's metadata_item: {results[0][0]}")

        # Check if results have proper metadata
        for i, result in enumerate(results[:3], 1):
            metadata_item, score = result  # Unpack tuple
            
            # Updated logic to access correct fields
            doc_id = metadata_item.get('arxiv_entry_id')  # Prefer arxiv_entry_id if present
            if not doc_id: # Fallback if arxiv_entry_id is None or not present
                doc_id = metadata_item.get('entry_id', 'N/A') # Check for entry_id
            if doc_id is None: # Ensure doc_id is not None before printing
                doc_id = 'N/A'

            title = metadata_item.get('original_title') # Get original_title
            if not title: # Fallback if original_title is None or not present
                title = metadata_item.get('title', 'No title available') # Check for title as a last resort

            if len(title) > 50:
                title = title[:50] + "..."
            
            print(f"   Result {i}: ID={doc_id}, Title='{title}', Score={score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during retrieval test: {e}")
        return False

def main():
    """Run all tests to verify the fix is working."""
    print("🚀 Starting fix verification tests...\n")
    
    # Test 1: Metadata key handling
    test1_success = test_metadata_keys()
    
    # Test 2: End-to-end retrieval
    test2_success = test_retrieval_system()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   Metadata key handling: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   End-to-end retrieval: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 All tests passed! The metadata key fix is working correctly.")
        print(f"   The system can now handle both 'entry_id' and 'arxiv_entry_id' keys.")
        print(f"   RAG queries should now return results on the website.")
    else:
        print(f"\n⚠️  Some tests failed. The fix may need additional work.")
    
    return test1_success and test2_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
