# -*- coding: utf-8 -*-
"""
Main script to run the CLI application with hybrid search and RAG.
Default mode: Strict RAG (answers only from local docs).
--general mode: Hybrid-Knowledge RAG (uses local docs + LLM internal knowledge).
"""

import argparse
import logging
import ssl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO messages
import sys
import arxiv # type: ignore
import nltk
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Generator, cast 
import textwrap
import traceback
import asyncio

# --- Logger Setup ---
# Configure logger specifically for this module
# Use basicConfig with force=True to ensure it applies even if root logger was configured elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-7s - [CLI] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
logger = logging.getLogger(__name__) # Use module name

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---


# --- Project Module Imports ---
try:
    from hybrid_search_rag import config
    # Ensure resource_fetcher is imported correctly
    from hybrid_search_rag.data_handling.resource_fetcher import (
        fetch_arxiv_papers,
        crawl_and_fetch_web_articles,
        PLAYWRIGHT_AVAILABLE, # Import for checking
        ResourceFetcher # Added ResourceFetcher
    )
    from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel
    from hybrid_search_rag.data_handling.data_manager import DataManager
    from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import HybridRecommender, NltkManager, RecommendationParams
    from rank_bm25 import BM25Okapi # type: ignore
    # Import the updated llm_interface
    from hybrid_search_rag.llm_services.llm_interface import get_llm_response, get_llm_response_stream
except ImportError as e:
    # Use print for critical import errors as logger might not be fully configured yet
    print(f"ERROR: Failed to import project modules in cli.py: {e}", file=sys.stderr)
    print(f"Ensure you are running from the project root or the path setup is correct.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)
# --- End Project Module Imports ---


# --- NLTK Data Check ---
# Flag to ensure check/download runs only once per script execution
_nltk_data_checked_cli = False

def check_nltk_data():
    """
    Checks required NLTK data ('punkt', 'stopwords').
    Attempts download if missing. Exits if download fails.
    Ensures check runs only once per script execution.
    """
    global _nltk_data_checked_cli
    if _nltk_data_checked_cli:
        # logger.debug("NLTK data check already performed in this CLI execution.")
        return # Already checked in this run

    required_data = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
    missing_data = []
    logger.info("Checking required NLTK data...")
    for name, path in required_data.items():
        try:
            nltk.data.find(path)
            logger.info(f"NLTK data '{name}' found.")
        except LookupError:
            logger.warning(f"NLTK data '{name}' not found.")
            missing_data.append(name) # Add to list if missing

    if missing_data:
        logger.warning(f"Missing NLTK data packages: {', '.join(missing_data)}.")
        # Use print for CLI visibility during download attempt
        print(f"\nAttempting to download missing NLTK data: {', '.join(missing_data)}...", file=sys.stderr) # Escaped newline
        download_success = True
        try:
            # Attempt to bypass SSL verification if needed (common issue)
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass # Doesn't exist, proceed normally
            else:
                ssl._create_default_https_context = _create_unverified_https_context # type: ignore
                logger.info("Applied SSL context workaround for NLTK download.")

            for item_name_to_download in missing_data: # Changed loop variable to avoid conflict
                print(f"Downloading NLTK package: {item_name_to_download}...")
                # Use quiet=False for CLI to show progress/errors
                if nltk.download(item_name_to_download, quiet=False): # Use item_name_to_download
                    logger.info(f"Successfully downloaded NLTK data '{item_name_to_download}'.")
                    # Verify immediately after download
                    try:
                        nltk.data.find(required_data[item_name_to_download]) # Use item_name_to_download for lookup
                        logger.info(f"Verified NLTK data '{item_name_to_download}' after download.")
                    except LookupError:
                        logger.error(f"Verification failed after downloading '{item_name_to_download}'. Download might be incomplete or corrupted.")
                        download_success = False
                        break # Stop trying if verification fails
                else:
                    # nltk.download returns None if download failed
                    logger.error(f"NLTK download command failed for '{item_name_to_download}'. Check network connection or NLTK server status.")
                    download_success = False
                    break # Stop trying if download fails

        except Exception as e:
            logger.error(f"An error occurred during NLTK download: {e}", exc_info=True)
            download_success = False

        if not download_success:
            print("Automatic NLTK download failed.", file=sys.stderr)
            print("Please try installing the data manually in your environment:", file=sys.stderr)
            print(">>> import nltk", file=sys.stderr)
            for name in missing_data: # Iterate using original 'name' from required_data if needed
                print(f">>> nltk.download('{name}')", file=sys.stderr)
            print("Exiting due to missing NLTK data.", file=sys.stderr)
            sys.exit(1) # Exit if essential data couldn't be obtained
        else:
            logger.info("Finished NLTK download attempt.")
    else:
        logger.info("All required NLTK data packages found.")

    _nltk_data_checked_cli = True # Mark as checked for this execution


# --- Text Chunking ---
def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Splits text into chunks by sentences with overlap."""
    if not text: return []
    try:
        # Ensure punkt is available before tokenizing
        nltk.data.find('tokenizers/punkt') # This will raise LookupError if check_nltk_data failed
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        logger.critical("NLTK 'punkt' tokenizer not found during chunking. Cannot proceed. Ensure NLTK data was downloaded.")
        # Re-raise or return empty list depending on desired handling
        raise # Let the calling function handle the critical error
    except Exception as e:
        logger.warning(f"Sentence tokenization failed with unexpected error: {e}. Falling back to newline split.")
        sentences = [p.strip() for p in text.split('\n') if p.strip()] # Corrected split character
        if not sentences: sentences = [p.strip() for p in text.split('.') if p.strip()] # Further fallback

    if not sentences:
        logger.warning("Could not extract sentences or lines for chunking.")
        return []

    chunks = []
    start_index = 0
    while start_index < len(sentences):
        end_index = min(start_index + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[start_index:end_index]
        chunks.append(" ".join(chunk_sentences))
        step = max(1, sentences_per_chunk - overlap_sentences)
        start_index += step
    return chunks
# --- End Text Chunking ---

# --- Global Variables for Loaded Components ---
loaded_metadata: Optional[List[Dict[str, Any]]] = None
loaded_embeddings: Optional[np.ndarray] = None
loaded_bm25_index: Optional[BM25Okapi] = None
loaded_recommender: Optional[HybridRecommender] = None

# --- Component Loading ---
def load_components(force_reload: bool = False): # Made sync - no async operations needed
    global loaded_metadata, loaded_embeddings, loaded_bm25_index, loaded_recommender
    if force_reload or loaded_metadata is None or loaded_recommender is None:
        logger.info(f"{'Forcing reload' if force_reload else 'Loading core components'}...")
        data_manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
        # Load data and cast bm25_index to its expected type
        meta_temp, embeddings_temp, bm25_index_object = data_manager.load_all_data() # Assuming this is sync
        loaded_metadata = meta_temp
        loaded_embeddings = embeddings_temp
        loaded_bm25_index = cast(Optional[BM25Okapi], bm25_index_object)

        if loaded_metadata is None:
            logger.error("Metadata loading failed. Cannot initialize recommender.")
            raise RuntimeError("Metadata loading failed. Run the 'fetch' command first.")

        try:
            # Ensure NLTK data needed by Recommender/NltkManager is checked/loaded
            check_nltk_data() # Assuming this is sync

            embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME) # Assuming this is sync
            loaded_recommender = HybridRecommender(embed_model=embedder) # Assuming this is sync
            logger.info("Recommender initialized.")
        except SystemExit: # Catch exit from check_nltk_data
             logger.critical("NLTK data check failed during component loading. Cannot initialize recommender.")
             raise RuntimeError("Failed to acquire NLTK data during component loading.")
        except Exception as e:
            logger.error(f"Recommender initialization failed: {e}", exc_info=True)
            loaded_recommender = None
            raise RuntimeError("Recommender initialization failed.") from e

        if loaded_embeddings is None: logger.warning("Embeddings file not found or invalid; semantic search disabled.")
        if loaded_bm25_index is None: logger.warning("BM25 index file not found or invalid; keyword search disabled.")
        if loaded_embeddings is None and loaded_bm25_index is None:
            logger.error("Both embeddings and BM25 index are missing. Search functionality severely limited.")

# --- Data Fetching Logic ---
async def setup_data_and_fetch(args: argparse.Namespace) -> str:
    """Handles the 'fetch' command: gets documents, chunks, embeds, indexes, and saves."""
    logger.info("Starting data fetch and processing...")
    status_messages = []

    # --- Ensure NLTK data is available before proceeding ---
    try:
        check_nltk_data() # Verify NLTK data specifically for the fetch process
    except SystemExit: # Catch exit if NLTK download fails
        return "Error: Failed to acquire necessary NLTK data ('punkt', 'stopwords'). Cannot proceed with fetch."
    # ---

    # Determine Sources
    arxiv_query = ""; target_urls: List[str] = []; max_results = args.num_arxiv
    try:
        if args.suggest_sources:
            logger.info(f"Attempting LLM source suggestion for topic: '{args.topic}'")
            if not args.topic: return "Error: Topic (-t) required for --suggest-sources."
            suggestion_prompt = f'''Suggest search sources for the topic "{args.topic}":
1. Two arXiv query strings. Prefix *exactly* with "ARXIV_QUERY: ".
2. Five high-quality, relevant URLs (e.g., conference proceedings, key labs, tech blogs). Prefix *exactly* with "URL: ".
Provide *only* the queries and URLs in the specified format.'''
            logger.info("Requesting source suggestions from LLM...")
            try:
                # Use the non-streaming interface for suggestions
                suggestion_response = get_llm_response(suggestion_prompt)
                if suggestion_response:
                    suggested_arxiv_queries = [line.replace("ARXIV_QUERY:", "").strip() for line in suggestion_response.splitlines() if line.strip().startswith("ARXIV_QUERY:")]
                    suggested_urls = [line.replace("URL:", "").strip() for line in suggestion_response.splitlines() if line.strip().startswith("URL:")]
                    arxiv_query = suggested_arxiv_queries[0] if suggested_arxiv_queries else ""
                    target_urls = suggested_urls if suggested_urls else []
                    if not arxiv_query: logger.warning("LLM did not suggest arXiv queries."); max_results = 0
                    if not target_urls: logger.warning("LLM did not suggest URLs.")
                    logger.info(f"Using LLM suggestions: arXiv='{arxiv_query}', URLs={len(target_urls)}")
                else:
                     logger.error("LLM suggestion failed (empty response). Clearing sources.")
                     arxiv_query = ""; max_results = 0; target_urls = []
                     status_messages.append("Warning: LLM suggestion failed.")
            except Exception as llm_e:
                 logger.error(f"Error during LLM suggestion API call: {llm_e}. Clearing sources.", exc_info=True)
                 arxiv_query = ""; max_results = 0; target_urls = []
                 status_messages.append(f"Warning: LLM suggestion failed ({llm_e}).")
        elif args.arxiv_query:
            arxiv_query = args.arxiv_query
            target_urls = config.TARGET_WEB_URLS if hasattr(config, 'TARGET_WEB_URLS') else [] # Use config or empty list
            logger.info(f"Using custom arXiv query: '{arxiv_query}' and {'default' if target_urls else 'no'} web URLs.")
        else:
            arxiv_query = config.DEFAULT_ARXIV_QUERY if hasattr(config, 'DEFAULT_ARXIV_QUERY') else ""
            target_urls = config.TARGET_WEB_URLS if hasattr(config, 'TARGET_WEB_URLS') else []
            logger.info("Using default arXiv query and web URLs from config (if defined).")
    except Exception as e:
         logger.error(f"Error determining sources: {e}", exc_info=True)
         return f"Error deciding sources: {e}"

    # Fetching Documents
    logger.info(f"Fetching: arXiv query='{arxiv_query}' (max={max_results}), URLs={len(target_urls)}")
    
    arxiv_metadata_list: List[Dict[str, Any]] = []
    web_metadata_list: List[Dict[str, Any]] = [] # Initialize web_metadata_list
    resource_fetcher_instance = None  # Initialize instance

    try:
        # Initialize ResourceFetcher
        # It will try to set up sync Playwright if use_playwright_for_pdfs is True (default)
        resource_fetcher_instance = ResourceFetcher(use_playwright_for_pdfs=True)

        # Enable PDF fetching for arXiv - now uses direct HTTP requests, no Playwright needed
        # The should_fetch_arxiv_pdfs variable is not used by fetch_arxiv_papers anymore.
        # fetch_arxiv_papers now only returns metadata including 'pdf_url'.
        # The actual PDF content fetching will be done later using the ResourceFetcher instance.
        logger.info("arXiv PDF metadata fetching enabled.")

        if arxiv_query and max_results > 0:
            try:
                # fetch_arxiv_papers now only gets metadata, not the PDF content directly.
                arxiv_metadata_list_with_potential_pdfs = await fetch_arxiv_papers(
                    query=arxiv_query,
                    max_results=max_results,
                    verbose=args.debug # Pass debug flag for verbose logging in fetcher
                )
                # Now, iterate and fetch PDF content if pdf_url is present
                processed_arxiv_docs = []
                for paper_meta in arxiv_metadata_list_with_potential_pdfs:
                    if paper_meta.get('pdf_url'):
                        logger.info(f"Attempting to fetch PDF content for: {paper_meta.get('title')} from {paper_meta.get('pdf_url')}")
                        # Use ResourceFetcher.fetch_document for PDF
                        # fetch_document is async, so await it.
                        # It handles the sync playwright call in a thread.
                        fetched_doc_data = await resource_fetcher_instance.fetch_document(
                            url=paper_meta['pdf_url'], 
                            source='arxiv', 
                            is_arxiv_pdf_link=True
                        )
                        if fetched_doc_data and fetched_doc_data.get('text'):
                            # Update the original metadata with the fetched text content
                            paper_meta['content'] = fetched_doc_data['text']
                            paper_meta['content_type'] = 'pdf' # Mark as PDF content
                            processed_arxiv_docs.append(paper_meta)
                            logger.info(f"Successfully fetched and processed PDF for: {paper_meta.get('title')}")
                        else:
                            logger.warning(f"Failed to fetch/process PDF for: {paper_meta.get('title')} from {paper_meta.get('pdf_url')}")
                            # Optionally, still add metadata even if PDF fetch failed, but without content
                            paper_meta['content'] = None # Ensure content is None
                            processed_arxiv_docs.append(paper_meta)
                    else:
                        logger.warning(f"No PDF URL for arXiv entry: {paper_meta.get('title')}. Skipping PDF fetch.")
                        paper_meta['content'] = None # Ensure content is None
                        processed_arxiv_docs.append(paper_meta)
                arxiv_metadata_list = processed_arxiv_docs

            except Exception as arxiv_e:
                logger.error(f"arXiv processing failed: {arxiv_e}", exc_info=True)
                status_messages.append(f"Warning: arXiv processing failed ({arxiv_e}).")
                arxiv_metadata_list = []
        else:
            arxiv_metadata_list = []

        # web_metadata_list: List[Dict[str, Any]] = [] # Moved initialization up
        if target_urls:
            try:
                # Assuming crawl_and_fetch_web_articles is updated to use ResourceFetcher internally
                # or that it returns metadata that then needs content fetching similar to arXiv.
                # For now, if it returns content directly, it's fine. If not, similar logic to above is needed.
                web_metadata_list = await crawl_and_fetch_web_articles(
                    start_urls=target_urls, 
                    process_pdfs_linked=True, 
                    max_pages_override=args.num_web_pages if hasattr(args, 'num_web_pages') else None
                )
            except Exception as web_e:
                 logger.error(f"Web crawling failed: {web_e}", exc_info=True)
                 status_messages.append(f"Warning: Web crawling failed ({web_e}).")
                 web_metadata_list = []
            logger.info(f"Web fetcher finished, got {len(web_metadata_list)} results.")
        else:
            web_metadata_list = []

    finally: # Ensure resource cleanup
        if resource_fetcher_instance:
            resource_fetcher_instance.close() # Close Playwright resources
        logger.info("Completed data fetching process.")

    original_documents = arxiv_metadata_list + web_metadata_list
    num_docs_fetched = len(original_documents)
    if num_docs_fetched == 0:
        if not status_messages:
            status_messages.append("No data fetched from any source.")
        return "Error: " + " ".join(status_messages)
        
    logger.info(f"Fetched {num_docs_fetched} total documents.")

    # Chunking Documents
    logger.info("Chunking documents...")
    all_chunk_metadata = []
    for doc_index, doc_meta in enumerate(original_documents):
        original_content = doc_meta.get('content', '') # content should now have PDF text
        original_url = doc_meta.get('url', f'doc_{doc_index}') # Use 'url' which is consistent
        
        # Ensure 'title' exists, provide a fallback
        original_title = doc_meta.get('title', 'Untitled Document')
        if not original_title or not isinstance(original_title, str):
            original_title = 'Untitled Document'


        if not original_content or not isinstance(original_content, str) or not original_content.strip():
            logger.warning(f"Document '{original_title}' (URL: {original_url}) has no content or invalid content type. Skipping chunking for this document.")
            continue
        try:
            text_chunks = chunk_text_by_sentences(original_content)
        except Exception as chunk_e:
             logger.error(f"Failed to chunk document {original_url}: {chunk_e}", exc_info=True)
             continue

        if not text_chunks:
            logger.warning(f"No chunks generated for document '{original_title}' (URL: {original_url}).")
            continue

        for chunk_index, chunk_text in enumerate(text_chunks):
            # Use 'entry_id' for arXiv, 'url' for web as the primary ID for the chunk
            # This ensures consistency with how recommender might expect IDs
            doc_id_for_chunk = doc_meta.get('entry_id') if doc_meta.get('source') == 'arxiv' else original_url

            all_chunk_metadata.append({
                "chunk_id": f"{doc_id_for_chunk}_chunk_{chunk_index}",
                "chunk_text": chunk_text,
                "original_url": original_url, # Keep the actual URL
                "original_title": original_title,
                "source": doc_meta.get('source', 'unknown'),
                "authors": doc_meta.get('authors', []),
                "published": doc_meta.get('published', None),
                # 'entry_id' will be the main ID for the chunk, derived from arXiv ID or URL
                "entry_id": f"{doc_id_for_chunk}_chunk_{chunk_index}"
            })
    num_chunks = len(all_chunk_metadata)
    if num_chunks == 0: return "Error: No text chunks generated from documents."
    logger.info(f"Generated {num_chunks} chunks.")

    # Processing & Saving Chunks
    logger.info("Processing chunks (embedding & indexing)...")
    manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
    embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    all_chunk_texts = [chunk['chunk_text'] for chunk in all_chunk_metadata]
    embeddings = None; bm25_index = None; embed_success = False; bm25_success = False

    try: # Embedding
        embeddings = embedder.encode(all_chunk_texts, task_type="RETRIEVAL_DOCUMENT")
        if embeddings is not None and isinstance(embeddings, np.ndarray) and embeddings.shape[0] == num_chunks:
            logger.info(f"Embeddings generated (Shape: {embeddings.shape}).")
            embed_success = True
        else: logger.error("Embedding generation failed or returned unexpected result.")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        status_messages.append("Warning: Embedding failed.")

    try: # BM25 Indexing
        # NltkManager should be ready now due to earlier check
        tokenized_corpus = [NltkManager.tokenize_text(t) for t in all_chunk_texts if isinstance(t, str) and t.strip()]
        tokenized_corpus = [tok for tok in tokenized_corpus if tok] # Remove empty lists after tokenization
        if tokenized_corpus:
            bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built.")
            bm25_success = True
        else:
            logger.warning("No valid tokens found for BM25 index after processing chunks.")
            if not NltkManager.NLTK_DATA_AVAILABLE['punkt']:
                 status_messages.append("Warning: BM25 indexing skipped (NLTK 'punkt' unavailable).")

    except Exception as e:
        logger.error(f"BM25 index building failed: {e}", exc_info=True)
        status_messages.append("Warning: BM25 indexing failed.")

    try: # Saving Data
        manager.save_all_data(all_chunk_metadata, embeddings, bm25_index)
        logger.info("Processed data saved successfully.")
        load_components(force_reload=True) # Reload components with new data
        success_msg = f"Success: Fetched {num_docs_fetched} docs, generated {num_chunks} chunks. Saved metadata"
        if embed_success: success_msg += ", embeddings"
        if bm25_success: success_msg += ", BM25 index"
        success_msg += "."
        if status_messages: success_msg += " " + " ".join(status_messages)
        return success_msg
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}", exc_info=True)
        return f"Error: Failed to save data. Details: {e}"

# --- Fallback Result Formatting ---
def format_fallback_results(results: List[Tuple[Dict[str, Any], float]], num_to_show: int) -> str:
    """Formats retrieval results into a string if the LLM fails."""
    if not results: return "LLM response failed; no relevant document chunks found."
    actual_num_to_show = min(num_to_show, len(results))
    lines = [f"LLM response failed. Top {actual_num_to_show} retrieved document chunks (fallback):"]
    for rank, (chunk_meta, score) in enumerate(results[:actual_num_to_show]):
        lines.append("-" * 20)
        lines.append(f"{rank + 1}. [Score: {score:.4f}] [Src: {chunk_meta.get('source', '?')}]")
        lines.append(f"   Title: {chunk_meta.get('original_title', 'N/A')}")
        lines.append(f"   URL: {chunk_meta.get('original_url', '#')}")
        lines.append(f"   Snippet: {(chunk_meta.get('chunk_text', '') or '')[:200]}...")
    return "\n".join(lines) # Changed to join with newline

# --- Recommendation Logic ---
async def run_recommendation(query: str, num_final_results: int, general_mode: bool, concise_mode: bool) -> Tuple[Generator[str, None, None], List[str], List[Dict[str, Any]]]:
    """
    Handles recommendation: retrieves chunks, prepares context, calls LLM stream.
    Returns a tuple: (response_generator, formatted_source_list, raw_context_chunks_list)
    """
    mode_name = "Hybrid" if general_mode else "Strict"
    style_name = "Concise" if concise_mode else "Detailed"
    logger.info(f"Running recommendation ({mode_name} RAG, {style_name} Prompt) for query: '{query[:100]}...'")

    response_generator: Generator[str, None, None]
    formatted_source_list: List[str] = []
    raw_context_chunks_list: List[Dict[str, Any]] = []
    hybrid_results: List[Tuple[Dict[str, Any], float]] = [] # Corrected type hint
    captured_exception = None

    try:
        if loaded_recommender is None or loaded_metadata is None:
             logger.info("Components not loaded, attempting load...")
             load_components()
        if loaded_recommender is None or loaded_metadata is None:
             raise RuntimeError("Core components failed to load. Run 'fetch' command first.")

        logger.info("Retrieving relevant document chunks...")
        num_candidates = max(config.RAG_NUM_DOCS + 5, num_final_results + 5)
        num_candidates = min(num_candidates, len(loaded_metadata))
        if num_candidates > 0:
            rec_params = RecommendationParams(
                semantic_candidates=config.SEMANTIC_CANDIDATES,
                keyword_candidates=config.KEYWORD_CANDIDATES,
                fusion_k=config.RANK_FUSION_K,
                top_n_final=num_candidates
            )
            # Ensure loaded_recommender is not None before calling recommend
            if loaded_recommender:
                hybrid_results = loaded_recommender.recommend(
                    query=query,
                    resource_metadata=loaded_metadata,
                    resource_embeddings=loaded_embeddings,
                    bm25_index=loaded_bm25_index,
                    params=rec_params
                )
            else: # Should be caught by earlier check, but as a safeguard
                raise RuntimeError("Recommender is None, cannot proceed with recommendation.")
            logger.info(f"Retrieved {len(hybrid_results)} candidate chunks.")
        else: logger.warning("Skipping retrieval (no metadata available or zero candidates needed).")

        context_string = "No relevant document chunks were found in the local data."
        reference_map = {}
        if hybrid_results:
            top_chunks_for_rag = hybrid_results[:config.RAG_NUM_DOCS]
            raw_context_chunks_list = [chunk_meta for chunk_meta, _ in top_chunks_for_rag]
            context_texts = []
            for i, chunk_meta in enumerate(raw_context_chunks_list):
                 chunk_num = i + 1
                 title = chunk_meta.get('original_title', 'N/A')
                 url = chunk_meta.get('original_url', '#')
                 snippet = (chunk_meta.get('chunk_text', '') or '')[:config.MAX_CONTEXT_LENGTH_PER_DOC]
                 context_texts.append(f"Source [{chunk_num}]:Title: {title} URL: {url} Content Snippet: {snippet}")
                 reference_map[chunk_num] = f"{title} (URL: {url})"
            context_string = "\n---\n".join(context_texts) # Use newline and --- for separation
            formatted_source_list = [f"[{num}] {details}" for num, details in reference_map.items()]
            logger.info(f"Prepared context from {len(raw_context_chunks_list)} chunks.")
        else:
            logger.info("No relevant document chunks found to provide as context.")
            formatted_source_list = ["No relevant document chunks found."]

        final_prompt: str
        prompt_template_base = """[INST] {system_message}

Provided Document Context:
{context}

User Question: {query} [/INST]
Answer:"""

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
        else: # Strict Mode
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
        final_prompt = prompt_template_base.format(system_message=system_message, context=context_string, query=query)
        logger.info("Preparing to call LLM for response generation...")
        response_generator = get_llm_response_stream(final_prompt)

    except RuntimeError as e:
        logger.error(f"Recommendation failed due to component loading error: {e}", exc_info=False)
        captured_exception = e
        def error_generator(err): yield f"Error: {err}. Cannot run recommendation."
        response_generator = error_generator(captured_exception)
        formatted_source_list = []
        raw_context_chunks_list = []
    except Exception as e:
        logger.error(f"Unexpected error during recommendation setup: {e}", exc_info=True)
        captured_exception = e
        def error_generator(err): yield f"An unexpected error occurred during setup: {err}"
        response_generator = error_generator(captured_exception)
        formatted_source_list = []
        raw_context_chunks_list = []

    return response_generator, formatted_source_list, raw_context_chunks_list


# --- arXiv Search Logic ---
def run_arxiv_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    """Performs a direct arXiv search."""
    logger.info(f"Searching arXiv directly for: '{query}' (Top {num_results})")
    results_list: List[Dict[str, Any]] = []
    
    client = arxiv.Client(page_size=min(num_results, 100), delay_seconds=1.0, num_retries=5)
    
    try:
        search = arxiv.Search(query=query, max_results=num_results, sort_by=arxiv.SortCriterion.Relevance)
        for result in client.results(search):
            results_list.append({
                "title": result.title or "N/A",
                "authors": [str(a) for a in result.authors],
                "published_date": result.published.strftime('%Y-%m-%d') if result.published else "N/A", # Changed key
                "entry_id": result.entry_id or 'N/A', # Added entry_id
                "pdf_url": result.pdf_url or 'N/A',
                "url": result.entry_id or 'N/A', # Use entry_id as primary URL
                "summary": (result.summary or '').replace('\n', ' ').strip()
            })
        logger.info(f"arXiv search found {len(results_list)} results.")
    except ssl.SSLError as ssl_err:
        logger.error(f"SSL error during arXiv search: {ssl_err}", exc_info=True)
        logger.info("SSL issue detected. Consider using the arxiv_ssl_patch.py script directly if issues persist or ensure system certificates are up to date.")
    except ConnectionError as conn_err:
        logger.error(f"Connection error during arXiv search: {conn_err}", exc_info=True)
    except Exception as e:
        logger.error(f"arXiv search failed: {e}", exc_info=True)
    return results_list


# --- Main CLI Execution ---
async def main_cli(): # Renamed to avoid conflict with main function in app.py
    parser = argparse.ArgumentParser(description='''CLI for Hybrid Search RAG.''')
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Fetch Command ---
    fetch_parser = subparsers.add_parser("fetch", help="Fetch and process data from sources.")
    fetch_parser.add_argument("--arxiv-query", type=str, default=config.DEFAULT_ARXIV_QUERY, help="Custom query for fetching arXiv papers.")
    fetch_parser.add_argument("--num-arxiv", type=int, default=config.MAX_ARXIV_RESULTS, help="Max number of arXiv papers to fetch.")
    fetch_parser.add_argument("--suggest-sources", action="store_true", help="Let LLM suggest arXiv query and web URLs based on a topic.")
    fetch_parser.add_argument("-t", "--topic", type=str, default="", help="Topic for LLM source suggestion (required if --suggest-sources).")
    fetch_parser.add_argument("--num-web-pages", type=int, default=config.MAX_PAGES_TO_CRAWL, help="Max number of web pages to crawl (used by crawl_and_fetch_web_articles).")


    # --- Recommend Command ---
    rec_parser = subparsers.add_parser("recommend", help="Get recommendations/answers for a query.")
    rec_parser.add_argument("query", type=str, help="The query to search for.")
    rec_parser.add_argument("-n", "--num-results", type=int, default=config.RAG_NUM_DOCS, help="Number of relevant context chunks to use for LLM.") # Changed help text
    rec_parser.add_argument("--general", action="store_true", help="Enable Hybrid RAG mode (LLM uses general knowledge + context). Default is Strict RAG.")
    rec_parser.add_argument("--concise", action="store_true", help="Use a concise prompt for the LLM, potentially faster but less conversational.")
    # Stream argument handled globally now

    # --- Find ArXiv Command ---
    find_arxiv_parser = subparsers.add_parser("find_arxiv", help="Directly search arXiv without RAG.")
    find_arxiv_parser.add_argument("query", type=str, help="The query for arXiv.")
    find_arxiv_parser.add_argument("-n", "--num-results", type=int, default=10, help="Number of arXiv results to display.")

    # --- Rebuild Index Command ---
    rebuild_parser = subparsers.add_parser("rebuild_index", help="Force rebuild of BM25 index and embeddings from existing metadata.")
    # No specific args for rebuild_index other than global ones

    # Global arguments
    parser.add_argument("--data-path", type=str, default=config.DATA_DIR, help="Path to the data directory.") # Kept for potential future use if config.DATA_DIR needs override
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")
    parser.add_argument("--stream", action='store_true', default=True, help="Stream LLM responses. Default: True.")
    parser.add_argument("--no-stream", dest='stream', action='store_false', help="Disable streaming LLM responses.")


    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled.")

    try:
        check_nltk_data()
    except SystemExit:
        print("Exiting: NLTK data setup failed.", file=sys.stderr)
        sys.exit(1)

    if args.command == "fetch":
        print("Fetching and processing data...")
        fetch_status = await setup_data_and_fetch(args)
        print(fetch_status)
        if fetch_status.startswith("Error:"):
            sys.exit(1)
        print("Data fetching and processing complete.")

    elif args.command == "rebuild_index":
        print("Rebuilding index and embeddings from existing data...")
        try:
            load_components(force_reload=False)
            if loaded_metadata:
                data_manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
                embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
                
                all_chunk_texts = [chunk['chunk_text'] for chunk in loaded_metadata]
                embeddings = None
                bm25_index = None
                embed_success = False
                bm25_success = False

                if not all_chunk_texts:
                    print("No chunk texts found in metadata. Cannot rebuild.")
                else:
                    try:
                        embeddings = embedder.encode(all_chunk_texts, task_type="RETRIEVAL_DOCUMENT")
                        if embeddings is not None and isinstance(embeddings, np.ndarray) and embeddings.shape[0] == len(all_chunk_texts):
                            logger.info(f"Embeddings regenerated (Shape: {embeddings.shape}).")
                            embed_success = True
                        else: logger.error("Embedding regeneration failed or returned unexpected result.")
                    except Exception as e:
                        logger.error(f"Embedding regeneration failed: {e}", exc_info=True)

                    try:
                        tokenized_corpus = [NltkManager.tokenize_text(t) for t in all_chunk_texts if isinstance(t, str) and t.strip()]
                        tokenized_corpus = [tok for tok in tokenized_corpus if tok] 
                        if tokenized_corpus:
                            bm25_index = BM25Okapi(tokenized_corpus)
                            logger.info("BM25 index rebuilt.")
                            bm25_success = True
                        else:
                            logger.warning("No valid tokens found for BM25 index from existing metadata.")
                    except Exception as e:
                        logger.error(f"BM25 index rebuilding failed: {e}", exc_info=True)

                    data_manager.save_all_data(loaded_metadata, embeddings, bm25_index)
                    load_components(force_reload=True)
                    print(f"Rebuild complete. Embeddings updated: {embed_success}. BM25 index updated: {bm25_success}.")
            else:
                print("No metadata loaded. Cannot rebuild. Run 'fetch' command first.")
        except RuntimeError as e:
            print(f"Error during rebuild: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "recommend":
        try:
            if loaded_recommender is None: load_components()
        except RuntimeError as e:
            # Keep this error print as it's a CLI operational error
            print(f"Error loading components for recommendation: {e}", file=sys.stderr)
            sys.exit(1)

        # Comment out or remove print statements related to displaying results in the CLI
        # print(f"Processing query: '{args.query}' (General: {args.general}, Concise: {args.concise}, Stream: {args.stream})")
        logger.info(f"CLI: Processing query: '{args.query}' (General: {args.general}, Concise: {args.concise}, Stream: {args.stream})") # Log instead of printing
        
        response_gen, sources, _ = await run_recommendation(args.query, args.num_results, args.general, args.concise)
        
        # full_response_parts = [] # Not needed if not printing
        if args.stream:
            # print("\\nLLM Response (Streaming):")
            try:
                for chunk in response_gen:
                    # print(chunk, end="", flush=True) # Suppress terminal output
                    # full_response_parts.append(chunk) # Not needed
                    pass # Consume the generator
                # print() 
            except Exception as e:
                # Keep this error print as it's a CLI operational error
                print(f"\\nError during streaming: {e}", file=sys.stderr) # Changed to sys.stderr
                logger.error(f"Error streaming LLM response: {e}", exc_info=True)
        else:
            # print("\\nLLM Response (Collected):")
            try:
                collected_response = "".join(list(response_gen))
                # print(collected_response) # Suppress terminal output
                # full_response_parts.append(collected_response) # Not needed
                pass # Consume the generator
            except Exception as e:
                # Keep this error print as it's a CLI operational error
                print(f"\\nError collecting non-streamed response: {e}", file=sys.stderr) # Changed to sys.stderr
                logger.error(f"Error collecting LLM response: {e}", exc_info=True)
        
        # print("\\n--- Sources ---")
        # if sources:
        #     for i, source_info in enumerate(sources):
        #         print(f"{i+1}. {source_info}")
        # else:
        #     print("No sources were provided with the response.")
        # print("---------------")
        logger.info("CLI: Recommendation processing complete. Results are intended for UI display.")


    elif args.command == "find_arxiv":
        results = run_arxiv_search(args.query, args.num_results)
        if results:
            print(f"\nFound {len(results)} results on arXiv for '{args.query}':")
            for i, paper in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"  Title: {paper.get('title', 'N/A')}")
                authors_str = ", ".join(paper.get('authors', [])) if paper.get('authors') else "N/A"
                print(f"  Authors: {authors_str}")
                print(f"  Published: {paper.get('published_date', 'N/A')}") # Key was changed
                print(f"  ID: {paper.get('entry_id', 'N/A')}") # Key was added
                print(f"  PDF URL: {paper.get('pdf_url', 'N/A')}")
                print(f"  Abstract: {textwrap.shorten(paper.get('summary', 'N/A'), width=150, placeholder='...')}")
        else:
            print(f"No results found on arXiv for '{args.query}'.")

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main_cli())
    except KeyboardInterrupt:
        print("\nCLI execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"A critical error occurred in the CLI: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

