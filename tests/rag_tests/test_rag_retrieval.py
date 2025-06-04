#!/usr/bin/env python3
"""
Test script to debug the RAG retrieval process.
"""

import sys
import os
import asyncio
import logging

# Add project root to path
project_root = r"c:/Users/felix/OneDrive/Documents/MachineLearningResearch-Assistant"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hybrid_search_rag.data_handling.data_manager import DataManager
from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import HybridRecommender
from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel
from hybrid_search_rag import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

async def test_rag_retrieval():
    """Test the RAG retrieval process step by step."""
    print("🧪 Testing RAG Retrieval Process")
    print("=" * 50)
    
    try:
        # 1. Load components
        print("📂 Loading components...")
        data_manager = DataManager(config.DATA_DIR)
        metadata, embeddings, bm25_index = data_manager.load_combined_data()
        
        print(f"✅ Loaded {len(metadata):,} documents")
        print(f"✅ Embeddings shape: {embeddings.shape}")
        print(f"✅ BM25 index loaded")
        
        # 2. Initialize embedder
        print("\n🤖 Initializing embedding model...")
        embedder = EmbeddingModel()
          # 3. Initialize recommender
        print("🔍 Initializing hybrid recommender...")
        recommender = HybridRecommender(embed_model=embedder, enable_query_preprocessing=True)
        
        # 4. Test queries
        test_queries = [
            "retrieval augmented generation",
            "RAG",
            "what is RAG",
            "machine learning",
            "neural networks",
            "transformer"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Testing query: '{query}'")
            
            try:
                # Get query embedding
                query_embedding = await embedder.get_embedding(query)
                print(f"   Query embedding shape: {query_embedding.shape}")
                
                # Test semantic search
                semantic_results = recommender._semantic_search(query_embedding, top_k=5)
                print(f"   Semantic results: {len(semantic_results)} items")
                
                # Test BM25 search
                bm25_results = recommender._bm25_search(query, top_k=5)
                print(f"   BM25 results: {len(bm25_results)} items")
                
                # Test hybrid search
                recommendations = await recommender.get_recommendations(query, num_docs=5)
                print(f"   Hybrid results: {len(recommendations.relevant_chunks)} chunks")
                
                if recommendations.relevant_chunks:
                    best_chunk = recommendations.relevant_chunks[0]
                    print(f"   Best match: {best_chunk.title[:60]}...")
                    print(f"   Score: {best_chunk.score:.4f}")
                else:
                    print("   ⚠️  No relevant chunks found!")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Failed to load components: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_retrieval())
