#!/usr/bin/env python3
"""
Test script to verify that the hybrid_search_rag package imports work correctly.
"""

def test_imports():
    """Test importing all main components from the package."""
    print("Testing imports from hybrid_search_rag package...")
    
    try:
        # Test core class imports
        from hybrid_search_rag import HybridRecommender, DataManager, EmbeddingModel
        print("✓ Core classes imported successfully")
        
        # Test LLM interface imports
        from hybrid_search_rag import get_llm_response, get_llm_response_stream
        print("✓ LLM interface functions imported successfully")
        
        # Test supporting classes
        from hybrid_search_rag import RecommendationParams
        print("✓ Supporting classes imported successfully")
        
        # Test data processing functions
        from hybrid_search_rag import combine_and_deduplicate_datasets
        print("✓ Data processing functions imported successfully")
        
        # Test version
        from hybrid_search_rag import __version__
        print(f"✓ Package version: {__version__}")
        
        print("\n🎉 All imports successful! The package structure is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_class_instantiation():
    """Test basic instantiation of main classes."""
    print("\nTesting class instantiation...")
    
    try:
        from hybrid_search_rag import EmbeddingModel, RecommendationParams
        
        # Test RecommendationParams (should work without dependencies)
        params = RecommendationParams(
            semantic_candidates=10,
            keyword_candidates=10,
            fusion_k=60,
            top_n_final=5
        )
        print("✓ RecommendationParams instantiated successfully")
        
        print("\n🎉 Basic instantiation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        test_class_instantiation()
    else:
        print("\n⚠️  Fix the import issues before proceeding.")
