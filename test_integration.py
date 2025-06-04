#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after the integration fixes.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all major imports."""
    try:
        print("Testing core imports...")
        from hybrid_search_rag import (
            HybridRecommender, 
            DataManager, 
            EmbeddingModel, 
            LLMInterface,
            get_llm_response, 
            get_llm_response_stream
        )
        print("✅ Core imports successful")
        
        print("\nTesting evaluation imports...")
        from hybrid_search_rag.evaluation import GenerationEvaluationMetrics
        print("✅ Evaluation imports successful")
        
        print("\nTesting component instantiation...")
        
        # Test LLMInterface
        llm_interface = LLMInterface()
        print("✅ LLMInterface instantiated")
        
        # Test evaluation metrics
        eval_metrics = GenerationEvaluationMetrics(llm_interface)
        print("✅ GenerationEvaluationMetrics instantiated")
        
        # Test that methods exist
        assert hasattr(eval_metrics, 'assess_faithfulness')
        assert hasattr(eval_metrics, 'assess_answer_relevance')
        assert hasattr(llm_interface, 'get_llm_response_unary')
        assert hasattr(llm_interface, 'get_llm_response_stream')
        print("✅ All required methods present")
        
        print("\n🎉 All integration tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_evaluation_functionality():
    """Test that the evaluation functionality works with the LLM interface."""
    try:
        print("\nTesting evaluation functionality...")
        
        from hybrid_search_rag import LLMInterface
        from hybrid_search_rag.evaluation import GenerationEvaluationMetrics
        
        # Initialize components
        llm_interface = LLMInterface()
        eval_metrics = GenerationEvaluationMetrics(llm_interface)
        
        # Test assessment methods (these might fail if no API keys are configured)
        print("Testing faithfulness assessment...")
        faith_result = eval_metrics.assess_faithfulness(
            generated_answer="The sky is blue due to Rayleigh scattering.",
            context="Sunlight reaches Earth's atmosphere and is scattered in all directions by all the gases and particles in the air. Blue light is scattered more than other colors because it travels as shorter, smaller waves. This is why we see a blue sky most of the time. This phenomenon is called Rayleigh scattering."
        )
        print(f"Faithfulness result: {faith_result}")
        
        print("Testing answer relevance assessment...")
        relevance_result = eval_metrics.assess_answer_relevance(
            query="Why is the sky blue?",
            generated_answer="The sky appears blue because of a phenomenon called Rayleigh scattering, where blue light from the sun is scattered more effectively by the Earth's atmosphere than other colors."
        )
        print(f"Relevance result: {relevance_result}")
        
        print("✅ Evaluation functionality test completed")
        return True
        
    except Exception as e:
        print(f"⚠️  Evaluation functionality test failed (expected if no API keys): {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Integration After Manual Edits ===")
    
    success = test_imports()
    if success:
        test_evaluation_functionality()
    
    print("\n=== Test Complete ===")
