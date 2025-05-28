#!/usr/bin/env python3
"""
Diagnostic script to understand why RAG is not finding relevant chunks.
"""

import json
import numpy as np
import os
from pathlib import Path

# Set up the path
data_dir = r"c:\Users\felix\OneDrive\Documents\MachineLearning\Research-Assistant\data_hybrid"

def diagnose_rag_data():
    """Diagnose the current state of RAG data."""
    print("🔍 RAG Data Diagnosis")
    print("=" * 50)
    
    # Check metadata
    metadata_path = os.path.join(data_dir, "combined_metadata.json")
    embeddings_path = os.path.join(data_dir, "combined_embeddings.npy")
    bm25_path = os.path.join(data_dir, "bm25_index.pkl")
    
    print("📁 File Status:")
    print(f"Metadata: {'✅' if os.path.exists(metadata_path) else '❌'} {metadata_path}")
    print(f"Embeddings: {'✅' if os.path.exists(embeddings_path) else '❌'} {embeddings_path}")
    print(f"BM25 Index: {'✅' if os.path.exists(bm25_path) else '❌'} {bm25_path}")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n📊 Metadata Analysis:")
            print(f"Total chunks: {len(data):,}")
            
            if data:
                sample = data[0]
                print(f"Sample chunk structure: {list(sample.keys())}")
                print(f"Sample title: {sample.get('original_title', 'N/A')}")
                print(f"Sample source: {sample.get('source', 'N/A')}")
                chunk_text = sample.get('chunk_text', '')
                print(f"Sample chunk length: {len(chunk_text)} characters")
                print(f"Sample chunk preview: {chunk_text[:200]}...")
                
                # Check for RAG-related terms
                rag_terms = ['retrieval augmented generation', 'rag', 'retrieval', 'generation']
                matching_chunks = 0
                for item in data[:1000]:  # Check first 1000 chunks
                    chunk_text = item.get('chunk_text', '').lower()
                    if any(term in chunk_text for term in rag_terms):
                        matching_chunks += 1
                
                print(f"\n🔍 Content Analysis (first 1000 chunks):")
                print(f"Chunks containing RAG-related terms: {matching_chunks}")
                
                # Check sources distribution
                sources = {}
                for item in data[:1000]:
                    source = item.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"\n📈 Source Distribution (first 1000 chunks):")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {source}: {count}")
            
        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
    
    if os.path.exists(embeddings_path):
        try:
            embeddings = np.load(embeddings_path)
            print(f"\n🔢 Embeddings Analysis:")
            print(f"Shape: {embeddings.shape}")
            print(f"Data type: {embeddings.dtype}")
            print(f"Memory usage: {embeddings.nbytes / (1024*1024):.2f} MB")
            
            # Check for NaN or infinite values
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            print(f"NaN values: {nan_count}")
            print(f"Infinite values: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                print("⚠️  Warning: Found NaN or infinite values in embeddings!")
                
        except Exception as e:
            print(f"❌ Error reading embeddings: {e}")

if __name__ == "__main__":
    diagnose_rag_data()
