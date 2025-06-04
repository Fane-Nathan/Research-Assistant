"""
Hybrid Search RAG Package

A production-ready Retrieval-Augmented Generation system that combines semantic search
with BM25 for improved document retrieval and question answering.

Main Components:
- HybridRecommender: Core search engine combining semantic and keyword search
- DataManager: Handles data loading, preprocessing, and storage
- EmbeddingModel: Manages text embeddings via Gemini API
- LLM Interface: Provides access to language model responses

Usage Example:
    from hybrid_search_rag import HybridRecommender, DataManager, EmbeddingModel
    
    # Initialize components
    embed_model = EmbeddingModel()
    data_manager = DataManager("data", "metadata.json", "embeddings.npy", "bm25_index.pkl")
    recommender = HybridRecommender(embed_model)
    
    # Load data and perform search
    data_manager.load_data()
    results = recommender.recommend("your query here", data_manager.get_documents())
"""

# Core Classes
from .retrieval_algorithm.hybrid_recommender import HybridRecommender, RecommendationParams
from .data_handling import DataManager, combine_and_deduplicate_datasets
from .embedding_services.gemini_embedder import EmbeddingModel

# LLM Interface Functions and Class
from .llm_services.llm_interface import get_llm_response, get_llm_response_stream, LLMInterface

# Evaluation Metrics
from .evaluation import GenerationEvaluationMetrics

__version__ = "1.0.0"

__all__ = [
    # Core Classes
    "HybridRecommender",
    "DataManager", 
    "EmbeddingModel",
    
    # LLM Interface
    "LLMInterface",
    "get_llm_response",
    "get_llm_response_stream",
    
    # Evaluation
    "GenerationEvaluationMetrics",
    
    # Supporting Classes
    "RecommendationParams",
    
    # Data Processing
    "combine_and_deduplicate_datasets",
]