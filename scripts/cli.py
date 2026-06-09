# -*- coding: utf-8 -*-
"""
Main script to run the CLI application with hybrid search and RAG.
This version has been corrected to align with the project's module structure
while preserving the original prompt engineering and logic.
"""

import argparse
import asyncio
import logging
import os
import ssl
import sys
import textwrap
import traceback
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import arxiv
import nltk
import numpy as np
from rank_bm25 import BM25Okapi

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Project Imports (Corrected) ---
try:
    from hybrid_search_rag import config
    from hybrid_search_rag.data_handling.data_manager import DataManager
    from hybrid_search_rag.data_handling.resource_fetcher import (
        ResourceFetcher,
        crawl_and_fetch_web_articles,
        fetch_arxiv_papers,
    )
    from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel
    from hybrid_search_rag.llm_services.llm_interface import (
        get_llm_response,
        get_llm_response_stream,
    )
    from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import (
        HybridRecommender,
        NltkManager,
        RecommendationParams,
    )
    from hybrid_search_rag.text_processing.text_cleaner import (
        clean_academic_title,
        clean_context_list,
    )
except ImportError as e:
    print(f"ERROR: Failed to import project modules in cli.py: {e}", file=sys.stderr)
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
)
logger = logging.getLogger("CLI")


def check_nltk_data() -> bool:
    """Checks for required NLTK data packages."""
    logger.info("Performing NLTK data check (CLI version)...")
    try:
        manager = NltkManager()
        if (manager.NLTK_DATA_AVAILABLE.get('punkt') or manager.NLTK_DATA_AVAILABLE.get('punkt_tab')) and manager.NLTK_DATA_AVAILABLE.get('stopwords'):
             logger.info("All required NLTK data packages appear to be available.")
             return True
        else:
            logger.error("NLTK data check failed. Manager initialized but required packages are missing.")
            return False
    except Exception as e:
        logger.error(f"An error occurred during NLTK check: {e}", exc_info=True)
        return False


def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Splits text into chunks by sentences with overlap."""
    if not text or not isinstance(text, str): return []
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        logger.warning(f"NLTK sentence tokenization failed: {e}. Falling back to newline split.")
        sentences = [p.strip() for p in text.split('\n') if p.strip()]

    if not sentences: return []

    chunks = []
    start_index = 0
    while start_index < len(sentences):
        end_index = min(start_index + sentences_per_chunk, len(sentences))
        chunks.append(" ".join(sentences[start_index:end_index]))
        start_index += max(1, sentences_per_chunk - overlap_sentences)
    return chunks

# --- Global Component Cache ---
_components_cache: Dict[str, Any] = {}

def load_components(force_reload: bool = False) -> Dict[str, Any]:
    """Loads all necessary RAG components, using an in-memory cache."""
    global _components_cache
    if not force_reload and _components_cache:
        logger.info("Returning cached components.")
        return _components_cache

    logger.info("Forcing reload..." if force_reload else "Loading components...")
    
    data_manager = DataManager(
        data_dir=config.DATA_DIR,
        metadata_filename=config.METADATA_FILE,
        embeddings_filename=config.EMBEDDINGS_FILE,
        bm25_filename=config.BM25_INDEX_FILE
    )
    metadata, embeddings, bm25_index_obj = data_manager.load_all_data()
    if metadata is None:
        metadata = []

    metadata_count = len(metadata)
    knowledge_base_is_empty = metadata_count == 0
    status_message = f"Loaded {metadata_count} metadata items. "

    nltk_manager = NltkManager()
    status_message += "NLTK data OK. "

    embed_model = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    recommender = HybridRecommender(embed_model=embed_model, enable_query_preprocessing=True)
    status_message += "Recommender initialized. "

    if embeddings is None: status_message += "Embeddings missing. "
    if bm25_index_obj is None: status_message += "BM25 index missing. "
    
    _components_cache = {
        "data_manager": data_manager,
        "recommender": recommender,
        "llm_interface": get_llm_response_stream,
        "knowledge_base_is_empty": knowledge_base_is_empty,
        "status_message": status_message,
        "nltk_manager": nltk_manager,
        "metadata": metadata,
        "embeddings": embeddings,
        "bm25_index": bm25_index_obj,
    }
    logger.info(f"load_components returning with status: {status_message}")
    return _components_cache


async def setup_data_and_fetch(args: argparse.Namespace) -> str:
    """Handles the 'fetch' command: gets documents, chunks, embeds, indexes, and saves."""
    if not check_nltk_data():
        return "Error: Critical NLTK data is missing. Cannot proceed."

    data_manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
    existing_metadata, _, _ = data_manager.load_all_data()
    existing_metadata = existing_metadata or []
    
    logger.info(f"Loaded {len(existing_metadata)} existing metadata items.")
    new_metadata_list: List[Dict[str, Any]] = []

    if hasattr(args, 'arxiv_query') and args.arxiv_query and args.num_arxiv > 0:
        papers = await fetch_arxiv_papers(query=args.arxiv_query, max_results=args.num_arxiv)
        for paper in papers:
            content = paper.get('summary', '')
            chunks = chunk_text_by_sentences(content)
            for i, chunk in enumerate(chunks):
                entry_id = paper.get('entry_id', '')
                new_metadata_list.append({
                    "chunk_id": f"{entry_id}_chunk_{i}",
                    "chunk_text": chunk, "original_url": paper.get('pdf_url', ''),
                    "original_title": paper.get('title', 'N/A'), "source": "arxiv",
                    "authors": paper.get('authors', []), "published": paper.get('published_date'),
                    "entry_id": f"{entry_id}_chunk_{i}"
                })

    if hasattr(args, 'web_urls') and args.web_urls:
        articles = await crawl_and_fetch_web_articles(args.web_urls)
        for article in articles:
            chunks = chunk_text_by_sentences(article.get('content', ''))
            for i, chunk in enumerate(chunks):
                url = article.get('url', '')
                new_metadata_list.append({
                    "chunk_id": f"{url}_chunk_{i}", "chunk_text": chunk,
                    "original_url": url, "original_title": article.get('title', 'N/A'), "source": "web",
                    "entry_id": f"{url}_chunk_{i}"
                })
    
    if not new_metadata_list: return "No new documents were fetched."

    seen_ids = {item.get("chunk_id") for item in existing_metadata}
    unique_new = [item for item in new_metadata_list if item.get("chunk_id") not in seen_ids]
    final_metadata = existing_metadata + unique_new
    
    logger.info(f"Processing {len(final_metadata)} total unique documents for indexing.")
    
    all_texts = [item.get('chunk_text', '') for item in final_metadata]
    embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(all_texts, task_type="RETRIEVAL_DOCUMENT")
    
    if embeddings is None or len(embeddings) != len(final_metadata):
        return "Error: Embedding generation failed or returned mismatched count."

    nltk_manager = NltkManager()
    tokenized_corpus = [nltk_manager.tokenize_text(text) for text in all_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    data_manager.save_all_data(final_metadata, embeddings, bm25_index)
    load_components(force_reload=True)
    return f"Success: Knowledge base updated. Total items: {len(final_metadata)}."


async def run_recommendation(
    query: str, num_final_results: int, general_mode: bool, concise_mode: bool
) -> Tuple[Optional[Generator[str, None, None]], List[str], List[Dict[str, Any]]]:
    """Handles recommendation: retrieves chunks, prepares context, calls LLM stream."""
    try:
        components = load_components()
        recommender: Optional[HybridRecommender] = components.get("recommender")
        metadata: Optional[List[Dict[str, Any]]] = components.get("metadata")
        embeddings: Optional[np.ndarray] = components.get("embeddings")
        bm25_index: Optional[BM25Okapi] = components.get("bm25_index")

        if recommender is None or metadata is None:
            raise RuntimeError("Core components (recommender/metadata) not loaded. Cannot generate recommendation.")

        params = RecommendationParams(
            semantic_candidates=config.SEMANTIC_CANDIDATES,
            keyword_candidates=config.KEYWORD_CANDIDATES,
            fusion_k=config.RANK_FUSION_K,
            top_n_final=num_final_results,
            expand_synonyms=True,
            top_n_rerank=5,
        )

        hybrid_results = recommender.recommend(
            query=query,
            resource_metadata=metadata,
            resource_embeddings=embeddings,
            bm25_index=bm25_index,
            params=params,
        )

        top_chunks = hybrid_results[: config.RAG_NUM_DOCS]
        raw_context_chunks_list = [chunk for chunk, score in top_chunks]
        context_string = "\n---\n".join(
            [
                f"Source [{i+1}]: {chunk.get('chunk_text', '')}"
                for i, (chunk, score) in enumerate(top_chunks)
            ]
        )

        prompt_template_base = "[INST] {system_message}\n\nProvided Document Context:\n{context}\n\nUser Question: {query} [/INST]\nAnswer:"

        if general_mode:
            if concise_mode:
                system_message = (
                    "You are a helpful AI research assistant."
                    "Provide a structured summary answering the user's question."
                    "Use the provided document context for key details, supplementing with your general knowledge where appropriate."
                    "**Format your entire response using standard Markdown.**"
                    "Use the following structure:"
                    "## Summary [Provide a brief overview answering the main question.]"
                    "## Key Details [List the most important findings or points using bullet points (* or -). Refer to information from the context.]"
                    "## Conclusion [Summarize the main takeaways.]"
                )
            else:
                system_message = (
                    "You are an AI assistant expert at analyzing technical documents."
                    "Provide a **detailed and comprehensive** answer to the user's question."
                    "Base your answer primarily on the 'Provided Document Context', but integrate relevant general knowledge smoothly for context and clarity."
                    "**Structure your response clearly using standard Markdown:**"
                    "**## Overview:** [Start with a concise paragraph summarizing the main answer.]"
                    "**## Detailed Analysis:** [Use ### Sub-Headings for distinct topics. Under each sub-heading, explain the topic thoroughly using multiple sentences or bullet points (* or -). Synthesize information across different context sources where applicable.]"
                    "**## Conclusion:** [Provide a concluding paragraph summarizing the key points and any limitations based on the provided context.]"
                )
        else:  # Strict Mode
            if concise_mode:
                system_message = (
                    "You are an AI assistant expert at analyzing technical documents."
                    "Provide a concise summary answering the user's question using **only** information found in the 'Provided Document Context'. **Do not use any outside knowledge.**"
                    "**Format your entire response using standard Markdown.**"
                    "Use the following structure:"
                    "## Summary [Provide a brief overview answering the main question based *only* on the context.]"
                    "## Key Findings [List the most important findings or points using bullet points (* or -). Extract information directly from the context.]"
                    "## Conclusion [Summarize the main takeaways based *only* on the context. State if the provided context is insufficient to fully answer.]"
                )
            else:
                system_message = (
                    "You are an AI assistant expert at analyzing technical documents."
                    "Provide a detailed answer using **only** information found in the 'Provided Document Context'. **Do not use any outside knowledge.**"
                    "**Structure your response clearly using standard Markdown:**"
                    "**## Overview:** [Start with a concise paragraph summarizing the main answer based *only* on the context.]"
                    "**## Detailed Analysis:** [Use ### Sub-Headings for distinct topics found *only* in the documents. Under each sub-heading, explain the topic thoroughly using multiple sentences or bullet points (* or -). Synthesize information across different context sources where applicable, but stick strictly to the provided text.]"
                    "**## Conclusion:** [Provide a concluding paragraph summarizing the key points based *only* on the context. Explicitly state if the information in the context is limited or insufficient.]"
                )

        final_prompt = prompt_template_base.format(
            system_message=system_message, context=context_string, query=query
        )

        response_generator = get_llm_response_stream(prompt=final_prompt)
        formatted_sources = [
            f"[{i+1}] {chunk.get('original_title', 'N/A')}"
            for i, (chunk, score) in enumerate(top_chunks)
        ]

        return response_generator, formatted_sources, raw_context_chunks_list
    except Exception as e:
        logger.error(f"Error in run_recommendation: {e}", exc_info=True)

        def error_gen():
            yield f"Error: {e}"

        return error_gen(), [], []


def run_arxiv_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    """Performs a direct search on arXiv and returns the results."""
    papers = asyncio.run(fetch_arxiv_papers(query, max_results=num_results))
    return [p for p in papers if isinstance(p, dict)]


async def main_cli():
    """Main asynchronous entry point for the CLI."""
    parser = argparse.ArgumentParser(description="CLI for Hybrid Search RAG.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and process data.")
    fetch_parser.add_argument("--arxiv-query", type=str, help="arXiv query.")
    fetch_parser.add_argument("--num-arxiv", type=int, default=10, help="Number of arXiv papers.")
    fetch_parser.add_argument("--web_urls", nargs="+", help="Web URLs to crawl.")

    rec_parser = subparsers.add_parser("recommend", help="Get recommendations.")
    rec_parser.add_argument("query", type=str, help="Search query.")
    rec_parser.add_argument("-n", "--num-results", type=int, default=5, help="Number of results.")
    rec_parser.add_argument("--general", action="store_true", help="Enable Hybrid RAG mode.")
    rec_parser.add_argument("--concise", action="store_true", help="Use a concise prompt.")

    args = parser.parse_args()

    if args.command == "fetch":
        status = await setup_data_and_fetch(args)
        print(status)
    elif args.command == "recommend":
        response_gen, sources, _ = await run_recommendation(
            args.query, args.num_results, args.general, args.concise
        )
        print("--- LLM Response ---")
        if response_gen:
            full_response = "".join(list(response_gen))
            print(full_response)
        print("\n--- Sources ---")
        print("\n".join(sources) if sources else "No sources found.")


if __name__ == "__main__":
    try:
        asyncio.run(main_cli())
    except KeyboardInterrupt:
        print("\nCLI execution interrupted by user.")
    except Exception as e:
        traceback.print_exc()
