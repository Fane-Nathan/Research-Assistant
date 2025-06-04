# hybrid_search_rag/evaluation/evaluator.py
"""
This script orchestrates the evaluation of the hybrid retrieval system.
It loads a predefined evaluation dataset (queries and ground truth relevance),
runs these queries through the recommender, and calculates various
information retrieval metrics using the `metrics.py` module.
"""
import json
import os
import logging
from typing import List, Dict, Any, Tuple, Set, Union, Optional, cast 
import numpy as np
from rank_bm25 import BM25Okapi

from .. import config 
from ..data_handling.data_manager import DataManager
from ..embedding_services.gemini_embedder import EmbeddingModel as GeminiEmbedder
from ..retrieval_algorithm.hybrid_recommender import HybridRecommender, RecommendationParams
from . import metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

EvaluationQuery = Dict[str, Any]

def load_evaluation_dataset(file_path: str) -> List[EvaluationQuery]:
    """
    Loads the evaluation dataset from a JSON file.
    The JSON file should be a list of objects, where each object represents
    a query and its ground truth relevance information.

    Expected format for each object in the JSON list:
    {
        "query_id": "q1",
        "query_text": "What is quantum entropy?",
        "relevant_doc_ids": ["docA", "docC", "docE"], // For P, R, AP
        "relevance_scores": { // For nDCG (doc_id: score)
            "docA": 3.0,
            "docC": 2.0,
            "docE": 3.0,
            "docB": 1.0, // Can include non-relevant retrieved items or other known relevances
            "docH": 2.0  // A relevant doc not necessarily in relevant_doc_ids for P/R/AP
        }
    }

    Args:
        file_path: Path to the JSON file containing the evaluation dataset.

    Returns:
        A list of EvaluationQuery objects.
    """
    if not os.path.exists(file_path):
        logger.error(f"Evaluation dataset file not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        logger.info(f"Successfully loaded {len(eval_data)} queries from {file_path}")
        for item in eval_data:
            if not all(key in item for key in ["query_id", "query_text", "relevant_doc_ids"]):
                logger.error(f"Invalid evaluation item missing required keys: {item}")
        return eval_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return []

def run_evaluation(
    evaluation_dataset_path: str,
    data_dir: str = config.DATA_DIR, # Use path from your config
    metadata_filename: str = config.METADATA_FILE,
    embeddings_filename: str = config.EMBEDDINGS_FILE,
    bm25_filename: str = config.BM25_INDEX_FILE,
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME, 
    k_values: List[int] = [5, 10, 20] # K values for P@K, R@K, nDCG@K
    ):
    """
    Runs the full evaluation pipeline.

    Args:
        evaluation_dataset_path: Path to the evaluation dataset JSON file.
        data_dir: Directory where corpus data (metadata, embeddings, bm25) is stored.
        metadata_filename: Name of the metadata file.
        embeddings_filename: Name of the embeddings file.
        bm25_filename: Name of the BM25 index file.
        embedding_model_name: Name of the embedding model to use (from config).
        k_values: A list of integers representing the 'K' for @K metrics.
    """
    logger.info("Starting evaluation pipeline...")

    eval_queries = load_evaluation_dataset(evaluation_dataset_path)
    if not eval_queries:
        logger.error("Evaluation cannot proceed: No evaluation queries loaded.")
        return

    logger.info("Initializing system components...")
    try:
        data_manager = DataManager(
            data_dir=data_dir,
            metadata_filename=metadata_filename,
            embeddings_filename=embeddings_filename,
            bm25_filename=bm25_filename
        )
        resource_metadata, resource_embeddings, loaded_bm25_index_obj = data_manager.load_all_data()

        bm25_index: Optional[BM25Okapi]
        if loaded_bm25_index_obj is None:
            bm25_index = None
        else:
            bm25_index = cast(BM25Okapi, loaded_bm25_index_obj)

        if resource_metadata is None:
            logger.error("Evaluation cannot proceed: Failed to load resource metadata.")
            return
        if resource_embeddings is None:
            logger.warning("Resource embeddings not loaded. Semantic search might be affected.")
        if bm25_index is None:
            logger.warning("BM25 index not loaded. Keyword search might be affected.")

        embed_model = GeminiEmbedder(model_name=embedding_model_name)

        recommender = HybridRecommender(embed_model=embed_model, enable_query_preprocessing=True)

        rec_params = RecommendationParams(
            semantic_candidates=config.SEMANTIC_CANDIDATES,
            keyword_candidates=config.KEYWORD_CANDIDATES,
            fusion_k=config.RANK_FUSION_K,
            top_n_final=max(k_values) + 20
        )
    except Exception as e:
        logger.error(f"Error during system component initialization: {e}", exc_info=True)
        return

    logger.info(f"Processing {len(eval_queries)} evaluation queries...")
    all_query_results = []

    aggregated_aps = []
    aggregated_metrics_at_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}

    for eval_query_item in eval_queries:
        query_id = eval_query_item["query_id"]
        query_text = eval_query_item["query_text"]
        true_relevant_set: Set[Union[str, int]] = set(eval_query_item["relevant_doc_ids"])
        true_relevance_map: Dict[Union[str, int], float] = eval_query_item.get("relevance_scores", {})

        logger.debug(f"Processing query_id: {query_id}, query_text: '{query_text}'")

        try:
            ranked_results_with_scores = recommender.recommend(
                query=query_text,
                resource_metadata=resource_metadata,
                resource_embeddings=resource_embeddings,
                bm25_index=bm25_index,
                params=rec_params
            )
            retrieved_doc_ids = []
            for result_tuple in ranked_results_with_scores:
                if isinstance(result_tuple, tuple) and \
                   len(result_tuple) > 0 and \
                   isinstance(result_tuple[0], dict) and \
                   'entry_id' in result_tuple[0]:
                    retrieved_doc_ids.append(result_tuple[0]['entry_id'])
                else:
                    logger.warning(f"Skipping malformed result tuple in ranked_results_with_scores for query {query_id}: {result_tuple}")
        
        except Exception as e:
            logger.error(f"Error recommending for query_id {query_id}: {e}", exc_info=True)
            retrieved_doc_ids = []
        
        logger.info(f"[Query: {query_id}] Retrieved {len(retrieved_doc_ids)} document IDs. First 5: {retrieved_doc_ids[:5]}")


        current_query_metrics = {"query_id": query_id, "query_text": query_text, "metrics": {}}

        ap = metrics.average_precision(retrieved_doc_ids, true_relevant_set)
        current_query_metrics["metrics"]["AP"] = ap
        if aggregated_aps is not None:
            aggregated_aps.append(ap)
        logger.debug(f"  [Query: {query_id}] AP: {ap:.4f}")

        for k_loop_var in k_values:
            logger.info(f"  [Query: {query_id}] Calculating P/R/nDCG@K for k_loop_var = {k_loop_var} (type: {type(k_loop_var)})")
            
            p_at_k_val: float = float('nan')
            r_at_k_val: float = float('nan')
            ndcg_val_k: float = float('nan')

            if not isinstance(k_loop_var, int) or k_loop_var <= 0:
                logger.error(f"  [Query: {query_id}] Invalid k_loop_var for metrics: {k_loop_var} (value or type). Skipping this k and assigning NaN.")
            else:
                p_at_k_val = metrics.precision_at_k(retrieved_doc_ids, true_relevant_set, k_loop_var)
                r_at_k_val = metrics.recall_at_k(retrieved_doc_ids, true_relevant_set, k_loop_var)

                if true_relevance_map:
                    ndcg_val_k = metrics.ndcg_at_k(retrieved_doc_ids, cast(Dict, true_relevance_map), k_loop_var)
                else:
                    logger.warning(f"  [Query: {query_id}] nDCG@{k_loop_var} not calculated for query {query_id} due to missing relevance_scores.")

            current_query_metrics["metrics"][f"P@{k_loop_var}"] = p_at_k_val
            current_query_metrics["metrics"][f"R@{k_loop_var}"] = r_at_k_val
            current_query_metrics["metrics"][f"nDCG@{k_loop_var}"] = ndcg_val_k

            if k_loop_var not in aggregated_metrics_at_k:
                 logger.error(f"Critical: k_loop_var={k_loop_var} not found in aggregated_metrics_at_k. This should not happen. Re-initializing.")
                 aggregated_metrics_at_k[k_loop_var] = {"precision": [], "recall": [], "ndcg": []}


            aggregated_metrics_at_k[k_loop_var]["precision"].append(p_at_k_val)
            aggregated_metrics_at_k[k_loop_var]["recall"].append(r_at_k_val)
            if not np.isnan(ndcg_val_k): 
                aggregated_metrics_at_k[k_loop_var]["ndcg"].append(ndcg_val_k)
            elif k_loop_var > 0:
                aggregated_metrics_at_k[k_loop_var]["ndcg"].append(ndcg_val_k)


            logger.debug(f"  [Query: {query_id}] P@{k_loop_var}: {p_at_k_val:.4f}, R@{k_loop_var}: {r_at_k_val:.4f}, nDCG@{k_loop_var}: {ndcg_val_k:.4f}")

        all_query_results.append(current_query_metrics)

    logger.info("\n--- Overall Evaluation Results ---")

    if aggregated_aps:
        mean_ap = metrics.mean_average_precision(aggregated_aps)
        logger.info(f"Mean Average Precision (MAP): {mean_ap:.4f}")
    else:
        logger.info("MAP: N/A (no queries processed or AP could not be calculated).")

    for k in k_values:
        mean_p_at_k = np.mean(aggregated_metrics_at_k[k]["precision"]) if aggregated_metrics_at_k[k]["precision"] else float('nan')
        mean_r_at_k = np.mean(aggregated_metrics_at_k[k]["recall"]) if aggregated_metrics_at_k[k]["recall"] else float('nan')
        mean_ndcg_at_k = np.mean(aggregated_metrics_at_k[k]["ndcg"]) if aggregated_metrics_at_k[k]["ndcg"] else float('nan')
        logger.info(f"For K={k}:")
        logger.info(f"  Mean Precision@{k}: {mean_p_at_k:.4f}")
        logger.info(f"  Mean Recall@{k}:    {mean_r_at_k:.4f}")
        logger.info(f"  Mean nDCG@{k}:      {mean_ndcg_at_k:.4f}")

    # Optionally, save detailed results per query to a file
    # output_results_path = os.path.join(config.PROJECT_ROOT, "evaluation_results_detailed.json") # Define in config
    # try:
    #     with open(output_results_path, 'w', encoding='utf-8') as f_out:
    #         json.dump(all_query_results, f_out, indent=2)
    #     logger.info(f"Detailed evaluation results saved to {output_results_path}")
    # except Exception as e:
    #     logger.error(f"Failed to save detailed evaluation results: {e}")

    logger.info("Evaluation pipeline finished.")


if __name__ == '__main__':
    # This is an example of how you might run the evaluation.
    # You'll need to create an 'evaluation_dataset.json' file.
    # Example evaluation_dataset.json content:
    # [
    #   {
    #     "query_id": "arxiv_q1",
    #     "query_text": "entropy and channel capacity",
    #     "relevant_doc_ids": ["0704.0046"],
    #     "relevance_scores": {
    #       "0704.0046": 3.0, /* This is the paper you showed earlier */
    #       "0704.0047": 0.0  /* Example of a non-relevant paper */
    #     }
    #   },
    #   {
    #     "query_id": "acoustic_q1",
    #     "query_text": "intelligent acoustic emission locator",
    #     "relevant_doc_ids": ["0704.0047"],
    #     "relevance_scores": {
    #       "0704.0047": 3.0,
    #       "0704.0046": 0.0
    #     }
    #   }
    # ]
    # Ensure the 'id' field in your actual resource_metadata matches the doc_ids here.

    # Construct the path to the evaluation dataset
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Goes up three levels
    eval_file_path = os.path.join(project_root, "data_store", "evaluation_sets", "evaluation_dataset_llm_labeled.json")

    if not os.path.exists(eval_file_path):
        logger.error(f"CRITICAL: Evaluation file {eval_file_path} not found. Please ensure it exists.")
    else:
        logger.info(f"Attempting to run evaluation with dataset: {eval_file_path}")
        run_evaluation(evaluation_dataset_path=eval_file_path)

