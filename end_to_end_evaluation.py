# run_end_to_end_evaluation.py
import logging
import numpy as np
from typing import Any, Dict, List, Set

# --- Import all necessary components ---
from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel
from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import (
    HybridRecommender,
    RecommendationParams,
)
from hybrid_search_rag.evaluation.retrieval_evaluator import RetrievalEvaluator
from rank_bm25 import BM25Okapi

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    An end-to-end script that prepares a raw dataset in memory and
    immediately runs a retrieval evaluation.
    """
    logging.info("--- Starting End-to-End In-Memory Evaluation ---")

    # =========================================================================
    # == Part 1: Data Preparation (In-Memory)
    # =========================================================================
    
    # The raw dataset you want to test.
    raw_data = [
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_0", "text": "1\nModular RAG: Transforming RAG Systems into\nLEGO-like Reconfigurable Frameworks..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_1", "text": "By decomposing complex RAG systems into independent modules and specialized operators..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_2", "text": "Finally, the paper explores the potential emergence of new operators and paradigms..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_3", "text": "Currently, RAG, as an enhancement method, has been widely applied in various practical application scenarios..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_4", "text": "Meng Wang and Haofen Wang are with College of Design and Innovation, Tongji University..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_5", "text": "The primary challenges of Naive RAG include: 1) Shallow Understanding of Queries..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_6", "text": "Feeding all retrieved chunks directly into LLMs is not always beneficial..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_7", "text": "For instance, query rewriting is used to make the queries more clear and specific..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_8", "text": "On the other hand, the growth in application demands has further propelled the evolution of RAG technology..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_9", "text": "This flexibility in the process significantly enhances the expressive power and adaptability of RAG systems..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_10", "text": "the expressive power and adaptability of RAG systems, enabling them to better adapt to various application scenarios..."},
        {"entry_id": "http://arxiv.org/abs/2407.21059v1_chunk_11", "text": "Access to heterogeneous data from multiple sources can provide the system with a richer knowledge background..."},
    ]

    # --- Process the raw data into the required formats ---
    
    resource_metadata = [{"entry_id": item["entry_id"], "text": item["text"]} for item in raw_data]
    texts_to_embed = [item["text"] for item in resource_metadata]
    
    logging.info(f"Processing {len(resource_metadata)} documents in memory...")

    embed_model = EmbeddingModel()
    resource_embeddings = embed_model.encode(texts=texts_to_embed)
    
    tokenized_corpus = [doc["text"].split(" ") for doc in resource_metadata]
    bm25_index = BM25Okapi(tokenized_corpus)

    logging.info("In-memory data preparation complete.")

    # =========================================================================
    # == Part 2: Evaluation
    # =========================================================================

    # --- <<< YOUR ACTION REQUIRED HERE >>> ---
    # Define your ground-truth evaluation set for the dataset above.
    evaluation_dataset = [
        {
            "query": "What are the limitations of Naive RAG?",
            "relevant_docs": {
                "http://arxiv.org/abs/2407.21059v1_chunk_4", # Mentions limitations of Naive RAG
                "http://arxiv.org/abs/2407.21059v1_chunk_5", # Details the challenges
            }
        },
        {
            "query": "What are the new challenges for modern RAG systems?",
            "relevant_docs": {
                "http://arxiv.org/abs/2407.21059v1_chunk_10", # Mentions complex data and orchestration
                "http://arxiv.org/abs/2407.21059v1_chunk_11", # Mentions new demands
            }
        },
    ]

    # --- Initialize components and set parameters ---
    recommender = HybridRecommender(embed_model)
    evaluator = RetrievalEvaluator(recommender)

    params = RecommendationParams(
        semantic_candidates=3, 
        keyword_candidates=3, 
        fusion_k=60,
        top_n_final=10
    )
    logging.info(f"Using recommendation parameters: {params}")

    # --- Run the evaluation using the in-memory artifacts ---
    logging.info("Starting the evaluation process...")
    evaluation_results = evaluator.evaluate(
        evaluation_dataset=evaluation_dataset,
        params=params,
        resource_metadata=resource_metadata,
        resource_embeddings=resource_embeddings,
        bm25_index=bm25_index
    )
    logging.info("Evaluation process completed.")

    # --- Print the final results ---
    evaluator.print_results(evaluation_results)

if __name__ == "__main__":
    main()