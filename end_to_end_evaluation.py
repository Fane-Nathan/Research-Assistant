import logging
import numpy as np
import json
from typing import Any, Dict, List, Set

# --- Import all necessary components ---
from hybrid_search_rag.retrieval_algorithm.reranker import ReRanker
from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel
from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import (
    HybridRecommender,
    RecommendationParams,
)
from hybrid_search_rag.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
    RetrievalEvaluationMetrics,
)

from rank_bm25 import BM25Okapi

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_evaluation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Loads the evaluation dataset from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Transform data to the expected format
        evaluation_dataset = [
            {
                "query": item["query_text"],
                "relevant_docs": set(item["relevant_doc_ids"]),
            }
            for item in data
        ]
        logging.info(f"Loaded {len(evaluation_dataset)} queries from {file_path}")
        return evaluation_dataset
    except FileNotFoundError:
        logging.error(f"Evaluation dataset not found at: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from: {file_path}")
        return []


def load_corpus(file_path: str) -> List[Dict[str, Any]]:
    """Loads the corpus from a JSON file and adapts its structure."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        
        # Adapt the structure of each item in the corpus
        for item in corpus:
            item['text'] = item.get('chunk_text', '')
            item['entry_id'] = item.get('chunk_id', '')

        logging.info(f"Loaded and adapted {len(corpus)} documents from {file_path}")
        return corpus
    except FileNotFoundError:
        logging.error(f"Corpus file not found at: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from: {file_path}")
        return []


def main():
    """
    An end-to-end script that prepares a raw dataset in memory and
    immediately runs a retrieval evaluation using a larger, file-based dataset.
    """
    logging.info("--- Starting End-to-End In-Memory Evaluation ---")

    # =========================================================================
    # == Part 1: Data Preparation (In-Memory)
    # =========================================================================

    # --- Load the corpus and evaluation dataset from files ---
    resource_metadata = load_corpus("data/combined_metadata.json")
    evaluation_dataset = load_evaluation_dataset(
        "data_store/evaluation_sets/evaluation_dataset_llm_labeled.json"
    )

    if not resource_metadata or not evaluation_dataset:
        logging.error("Could not load necessary data. Aborting evaluation.")
        return

    texts_to_embed = [item["text"] for item in resource_metadata]

    logging.info(f"Processing {len(resource_metadata)} documents in memory...")

    embed_model = EmbeddingModel()
    resource_embeddings = embed_model.encode(texts=texts_to_embed, task_type="retrieval_document")

    tokenized_corpus = [doc["text"].split(" ") for doc in resource_metadata]
    bm25_index = BM25Okapi(tokenized_corpus)

    logging.info("In-memory data preparation complete.")

    # =========================================================================
    # == Part 2: Evaluation with Re-ranking
    # =========================================================================

    # --- Initialize all components ---
    recommender = HybridRecommender(embed_model)
    reranker = ReRanker()
    evaluator = RetrievalEvaluator(recommender)

    # --- Set parameters for the full pipeline ---
    params = RecommendationParams(
        semantic_candidates=50,
        keyword_candidates=50,
        fusion_k=60,
        top_n_final=100,  # Stage 1: Retrieve more candidates for higher recall
        expand_synonyms=True,
        top_n_rerank=10,  # Stage 2: Return the final top 10 after re-ranking
    )
    logging.info(f"Using full pipeline parameters: {params}")

    final_metrics = RetrievalEvaluationMetrics()

    for item in evaluation_dataset:
        query = item["query"]
        relevant_docs_set = item["relevant_docs"]

        # STAGE 1: RETRIEVE a large, noisy set of candidates
        candidate_results = recommender.recommend(
            query=query,
            params=params,
            resource_metadata=resource_metadata,
            resource_embeddings=resource_embeddings,
            bm25_index=bm25_index,
        )

        candidate_docs = [doc for doc, score in candidate_results]

        # STAGE 2: RE-RANK the candidates for high precision
        reranked_results = reranker.rerank(query, candidate_docs)

        # Take the top N results after re-ranking
        final_recommendations = reranked_results[: params.top_n_rerank]

        # --- Evaluate the final, re-ranked results ---
        retrieved_ids = [doc.get('entry_id') for doc, score in final_recommendations]

        (
            hit_rate,
            mrr,
            ndcg,
        ) = evaluator._calculate_ranking_metrics(retrieved_ids, relevant_docs_set)
        final_metrics.update_ranking_metrics(hit_rate, mrr, ndcg)

        (
            precision,
            recall,
            f1,
        ) = evaluator._calculate_classification_metrics(retrieved_ids, relevant_docs_set)
        final_metrics.update_classification_metrics(precision, recall, f1)

    logging.info("Evaluation process completed.")
    evaluator.print_results(final_metrics)


if __name__ == "__main__":
    main()