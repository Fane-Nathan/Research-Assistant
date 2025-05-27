# scripts/bootstrap_evaluation_set.py
"""
This script helps bootstrap the creation of an evaluation dataset.
It generates candidate queries (e.g., from document titles or text snippets)
and uses the HybridRecommender to retrieve an initial set of potentially
relevant documents for each query.

The output of this script is a JSON file that the user can then review
manually to select truly relevant documents and assign relevance scores,
forming the basis of the final 'evaluation_dataset.json'.
"""
import sys
import os
import json
import random
import logging
from typing import List, Dict, Any, Optional, cast

# --- Path Setup ---
# Add the project root directory (the one containing 'hybrid_search_rag' and 'scripts')
# to the Python path. This allows us to import modules from the 'hybrid_search_rag'
# package when running this script from within the 'scripts' directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

from rank_bm25 import BM25Okapi

from hybrid_search_rag import config
from hybrid_search_rag.data_handling.data_manager import DataManager
from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel as GeminiEmbedder
from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import HybridRecommender, RecommendationParams

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def generate_candidate_queries_from_metadata(
    resource_metadata: List[Dict[str, Any]],
    num_queries_to_generate: int,
    min_title_length: int = 5,
    use_snippet_fallback: bool = True,
    snippet_length: int = 15
) -> List[Dict[str, str]]:
    """
    Generates candidate queries from existing resource metadata.
    It prioritizes 'original_title' and can fall back to 'chunk_text' snippets.
    """
    candidate_queries: List[Dict[str, str]] = []
    potential_sources: List[Dict[str, str]] = []
    logger.info(f"Attempting to generate queries from {len(resource_metadata)} metadata items.")

    for i, doc_meta in enumerate(resource_metadata):
        title = doc_meta.get("original_title", "")
        chunk_text = doc_meta.get("chunk_text", "")
        # Use chunk_id if available, otherwise fall back to id, then a generated index
        source_doc_id = doc_meta.get("chunk_id", doc_meta.get("id", f"doc_idx_{i}"))
        query_text_candidate: Optional[str] = None

        if title and len(title.split()) >= min_title_length:
            query_text_candidate = title
        elif use_snippet_fallback and chunk_text:
            # Take the first `snippet_length` words for the snippet
            snippet = " ".join(chunk_text.split()[:snippet_length])
            if snippet and len(snippet.split()) >= min_title_length: # Check if snippet is also valid
                query_text_candidate = snippet

        if query_text_candidate:
            potential_sources.append({
                "query_text": query_text_candidate.strip(),
                "source_doc_id": str(source_doc_id) # Ensure source_doc_id is a string
            })

    if not potential_sources:
        logger.warning("No suitable titles or text snippets found in metadata to generate any potential queries.")
        return []

    num_to_select = min(num_queries_to_generate, len(potential_sources))
    if num_to_select == 0: # Should be caught by `not potential_sources` but good for clarity
        logger.warning("No queries selected after filtering potential sources (num_to_select is 0).")
        return []
    if num_to_select < num_queries_to_generate:
        logger.warning(f"Could only select {num_to_select} queries, less than requested {num_queries_to_generate}.")

    # Randomly sample from potential sources to get variety
    selected_sources = random.sample(potential_sources, num_to_select)
    for i, src in enumerate(selected_sources):
        # Sanitize source_doc_id for use in query_id (replace common problematic chars)
        sanitized_source_doc_id = src['source_doc_id'].replace('/', '_').replace(':', '_').replace('.', '_')
        candidate_queries.append({
            "query_id": f"auto_q_{i+1}_from_{sanitized_source_doc_id[:30]}", # Truncate long IDs
            "query_text": src["query_text"]
        })
    logger.info(f"Successfully generated {len(candidate_queries)} candidate queries.")
    return candidate_queries


def bootstrap_evaluation_candidates(
    output_review_file_path: str,
    num_candidate_queries: int = 20,
    num_docs_per_query: int = 20,
    data_directory: str = config.DATA_DIR,
    metadata_filename: str = config.METADATA_FILE,
    embeddings_filename: str = config.EMBEDDINGS_FILE,
    bm25_filename: str = config.BM25_INDEX_FILE,
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME
):
    """
    Main function to generate a file with candidate queries and their
    top retrieved documents for manual review.
    """
    logger.info("Starting evaluation dataset bootstrapping process...")

    # 1. Load existing corpus data
    logger.info(f"Loading corpus data from directory: {data_directory}")
    data_manager = DataManager(
        data_dir=data_directory,
        metadata_filename=metadata_filename,
        embeddings_filename=embeddings_filename,
        bm25_filename=bm25_filename
    )
    resource_metadata, resource_embeddings, loaded_bm25_index_obj = data_manager.load_all_data()

    if not resource_metadata:
        logger.error("Failed to load resource metadata. Cannot proceed with bootstrapping.")
        return

    # Cast the loaded BM25 index object to the expected type for the recommender
    typed_bm25_index: Optional[BM25Okapi] = None
    if loaded_bm25_index_obj is not None:
        if isinstance(loaded_bm25_index_obj, BM25Okapi):
            typed_bm25_index = loaded_bm25_index_obj
        else:
            logger.warning(f"Loaded BM25 index is of type '{type(loaded_bm25_index_obj)}', not BM25Okapi. Keyword search may not work as expected.")
            # Depending on strictness, you might choose to set it to None or attempt casting
            try:
                typed_bm25_index = cast(BM25Okapi, loaded_bm25_index_obj) # Inform Pylance
            except Exception: # Should not happen if it's already the wrong type
                 logger.error("Failed to cast loaded BM25 object to BM25Okapi.")
                 typed_bm25_index = None # Fallback
    else:
        logger.info("No BM25 index loaded (file might be missing or empty).")


    # 2. Generate candidate queries
    logger.info(f"Generating {num_candidate_queries} candidate queries...")
    candidate_queries = generate_candidate_queries_from_metadata(resource_metadata, num_candidate_queries)
    if not candidate_queries:
        logger.error("No candidate queries were generated. Exiting bootstrapping process.")
        return

    # 3. Initialize Recommender
    logger.info("Initializing recommender system...")
    try:
        # Check for API key if using a cloud-based embedder like Gemini
        if "gemini" in embedding_model_name.lower() and not config.GOOGLE_API_KEY:
             logger.warning("GOOGLE_API_KEY might not be set in config; GeminiEmbedder initialization might fail or use defaults if any.")

        embed_model = GeminiEmbedder(model_name=embedding_model_name)
        recommender = HybridRecommender(embed_model=embed_model)
        
        rec_params = RecommendationParams(
            semantic_candidates=max(50, num_docs_per_query + 30), # Fetch more candidates than needed
            keyword_candidates=max(50, num_docs_per_query + 30),   # for better fusion
            fusion_k=config.RANK_FUSION_K, # RRF k parameter from config
            top_n_final=num_docs_per_query # Final number of documents to suggest per query
        )
    except Exception as e:
        logger.error(f"Error initializing recommender components: {e}", exc_info=True)
        return

    # 4. For each query, get candidates and format for review
    output_for_review: List[Dict[str, Any]] = []
    logger.info(f"Retrieving candidates for {len(candidate_queries)} queries...")

    for cand_query in candidate_queries:
        query_id = cand_query["query_id"]
        query_text = cand_query["query_text"]
        logger.debug(f"Processing candidate query: {query_id} - '{query_text[:50]}...'")

        try:
            # Pass the correctly typed (or cast) bm25_index
            ranked_results_with_scores = recommender.recommend(
                query=query_text,
                resource_metadata=resource_metadata,
                resource_embeddings=resource_embeddings,
                bm25_index=typed_bm25_index, # Use the casted/validated variable
                params=rec_params
            )

            candidates_for_this_query: List[Dict[str, Any]] = []
            for res_meta, score in ranked_results_with_scores:
                # Ensure 'id' or 'chunk_id' exists, provide a fallback
                doc_id_val = res_meta.get("chunk_id", res_meta.get("id", f"N/A_ID_for_{query_id}"))
                doc_title_val = res_meta.get("original_title", "N/A_TITLE")
                doc_chunk_text_val = res_meta.get("chunk_text", "") # Default to empty string

                candidates_for_this_query.append({
                    "doc_id": str(doc_id_val), # Ensure doc_id is string
                    "doc_title": str(doc_title_val),
                    "doc_snippet": (doc_chunk_text_val[:250] + "...") if doc_chunk_text_val else "N/A",
                    "retrieval_score": float(score) # Ensure score is float
                })

            output_for_review.append({
                "query_id": query_id,
                "query_text": query_text,
                "suggested_candidates_for_review": candidates_for_this_query
            })
        except Exception as e:
            logger.error(f"Error retrieving candidates for query '{query_text}': {e}", exc_info=True)
            # Add query with error information to the output for review
            output_for_review.append({
                "query_id": query_id,
                "query_text": query_text,
                "suggested_candidates_for_review": [],
                "error": str(e)
            })

    # 5. Save the output for review
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_review_file_path)
        if output_dir: # Check if output_dir is not empty (i.e., not saving to current dir)
            os.makedirs(output_dir, exist_ok=True)

        with open(output_review_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_for_review, f, indent=2)
        logger.info(f"Successfully saved candidate queries and documents for review to: {output_review_file_path}")
        logger.info("NEXT STEP: Manually review this file. For each query, identify relevant documents from its 'suggested_candidates_for_review'.")
        logger.info("Then, create your 'evaluation_dataset.json' using the format expected by 'evaluator.py'.")
    except IOError as e:
        logger.error(f"IOError saving the review file to {output_review_file_path}: {e}", exc_info=True)
    except Exception as e: # Catch other potential errors during save
        logger.error(f"Unexpected error saving the review file to {output_review_file_path}: {e}", exc_info=True)


if __name__ == '__main__':
    # Define the output path for the review file, placing it in a structured location
    review_file_path = os.path.join(project_root, "data_store", "evaluation_sets", "candidates_for_review.json")
    
    NUM_QUERIES_TO_GENERATE = 25 # Number of candidate queries to generate
    NUM_CANDIDATES_PER_QUERY_FOR_REVIEW = 15 # Number of docs to retrieve for each query

    # Ensure config.DATA_DIR is set and valid before proceeding
    if not config.DATA_DIR or not os.path.isdir(config.DATA_DIR): 
        logger.error(f"config.DATA_DIR ('{config.DATA_DIR}') is not set or is not a valid directory. Please configure it in config.py.")
    else:
        logger.info(f"Using data directory from config: {config.DATA_DIR}")
        bootstrap_evaluation_candidates(
            output_review_file_path=review_file_path,
            num_candidate_queries=NUM_QUERIES_TO_GENERATE,
            num_docs_per_query=NUM_CANDIDATES_PER_QUERY_FOR_REVIEW
            # data_directory parameter defaults to config.DATA_DIR
        )

