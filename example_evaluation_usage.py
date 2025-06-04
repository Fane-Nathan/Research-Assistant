#!/usr/bin/env python3
"""
Example usage of the RAG system evaluation functionality.

This script demonstrates how to use the GenerationEvaluationMetrics class
to assess the quality of RAG system outputs.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Example usage of evaluation functionality."""
    
    # Import the required components
    from hybrid_search_rag import (
        LLMInterface, 
        GenerationEvaluationMetrics,
        HybridRecommender,
        DataManager,
        EmbeddingModel
    )
    
    print("=== RAG System Evaluation Example ===\n")
    
    # 1. Initialize the LLM interface and evaluation metrics
    print("1. Initializing evaluation components...")
    llm_interface = LLMInterface()
    eval_metrics = GenerationEvaluationMetrics(llm_interface)
    print("✅ Evaluation components ready\n")
    
    # 2. Example evaluation scenarios
    print("2. Running evaluation examples...\n")
    
    # Example 1: Faithfulness Assessment
    print("Example 1: Faithfulness Assessment")
    print("-" * 40)
    
    context = """
    The greenhouse effect is a natural process that warms the Earth's surface. 
    When the Sun's energy reaches the Earth's atmosphere, some of it is reflected 
    back to space and the rest is absorbed and re-radiated by greenhouse gases. 
    The absorbed energy warms the atmosphere and the surface of the Earth.
    """
    
    generated_answer = "The greenhouse effect is a process where greenhouse gases trap heat in Earth's atmosphere, warming the planet."
    
    faith_result = eval_metrics.assess_faithfulness(
        generated_answer=generated_answer,
        context=context.strip()
    )
    
    print(f"Generated Answer: {generated_answer}")
    print(f"Faithfulness Rating: {faith_result['rating_str']}")
    print(f"Faithfulness Score: {faith_result['score']}")
    if 'error' in faith_result:
        print(f"Error: {faith_result['error']}")
    print()
    
    # Example 2: Answer Relevance Assessment
    print("Example 2: Answer Relevance Assessment")
    print("-" * 40)
    
    query = "What causes climate change?"
    answer = "Climate change is primarily caused by increased concentrations of greenhouse gases in the atmosphere, mainly from human activities like burning fossil fuels."
    
    relevance_result = eval_metrics.assess_answer_relevance(
        query=query,
        generated_answer=answer
    )
    
    print(f"Query: {query}")
    print(f"Generated Answer: {answer}")
    print(f"Relevance Rating: {relevance_result['rating_str']}")
    print(f"Relevance Score: {relevance_result['score']}")
    if 'error' in relevance_result:
        print(f"Error: {relevance_result['error']}")
    print()
    
    # 3. Integration with RAG system (example workflow)
    print("3. Example RAG System Integration")
    print("-" * 40)
    print("Note: This shows how evaluation would integrate with the full RAG pipeline:")
    print()
    print("# Initialize RAG components")
    print("embed_model = EmbeddingModel()")
    print("data_manager = DataManager('data', 'metadata.json', 'embeddings.npy', 'bm25_index.pkl')")
    print("recommender = HybridRecommender(embed_model)")
    print()
    print("# Load data and perform search")
    print("data_manager.load_all_data()  # Load processed data")
    print("search_results = recommender.recommend('query', data_manager.get_documents())")
    print()
    print("# Generate answer using LLM (you would implement this)")
    print("# generated_answer = generate_answer_from_context(query, search_results)")
    print()
    print("# Evaluate the generated answer")
    print("faith_score = eval_metrics.assess_faithfulness(generated_answer, search_results)")
    print("relevance_score = eval_metrics.assess_answer_relevance(query, generated_answer)")
    print()
    
    print("✅ Example complete!")
    print("\nThe evaluation system is ready for production use!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running example: {e}")
        print("This might be due to missing API keys or configuration issues.")
