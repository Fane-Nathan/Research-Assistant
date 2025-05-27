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
import arxiv
import nltk
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Generator, cast # Added Generator
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
    from hybrid_search_rag.data_handling.resource_fetcher import fetch_arxiv_papers, crawl_and_fetch_web_articles
    from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel
    from hybrid_search_rag.data_handling.data_manager import DataManager
    from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import HybridRecommender, NltkManager, RecommendationParams
    from rank_bm25 import BM25Okapi
    # Import the updated llm_interface
    from hybrid_search_rag.llm_services.llm_interface import get_llm_response, get_llm_response_stream
    # Highlighting is assumed removed based on previous steps
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
        print(f"\\nAttempting to download missing NLTK data: {', '.join(missing_data)}...", file=sys.stderr) # Escaped newline
        download_success = True
        try:
            # Attempt to bypass SSL verification if needed (common issue)
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass # Doesn't exist, proceed normally
            else:
                ssl._create_default_https_context = _create_unverified_https_context
                logger.info("Applied SSL context workaround for NLTK download.")

            for item_name in missing_data: # Changed loop variable to avoid conflict
                print(f"Downloading NLTK package: {item_name}...")
                # Use quiet=False for CLI to show progress/errors
                if nltk.download(item_name, quiet=False): # Use item_name
                    logger.info(f"Successfully downloaded NLTK data '{item_name}'.")
                    # Verify immediately after download
                    try:
                        nltk.data.find(required_data[item_name]) # Use item_name for lookup
                        logger.info(f"Verified NLTK data '{item_name}' after download.")
                    except LookupError:
                        logger.error(f"Verification failed after downloading '{item_name}'. Download might be incomplete or corrupted.")
                        download_success = False
                        break # Stop trying if verification fails
                else:
                    # nltk.download returns None if download failed
                    logger.error(f"NLTK download command failed for '{item_name}'. Check network connection or NLTK server status.")
                    download_success = False
                    break # Stop trying if download fails

        except Exception as e:
            logger.error(f"An error occurred during NLTK download: {e}", exc_info=True)
            download_success = False

        if not download_success:
            print("Automatic NLTK download failed.", file=sys.stderr)
            print("Please try installing the data manually in your environment:", file=sys.stderr)
            print(">>> import nltk", file=sys.stderr)
            for name in missing_data:
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
        sentences = [p.strip() for p in text.split('') if p.strip()]
        if not sentences: sentences = [p.strip() for p in text.split('') if p.strip()]

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
async def setup_data_and_fetch(args: argparse.Namespace) -> str: # Changed to async
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
    arxiv_query = ""; target_urls: List[str] = []; max_results = args.num_arxiv # Added type hint for target_urls
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
    logger.info(f"Fetching: arXiv query='{arxiv_query}' (max={max_results}), URLs={len(target_urls)}") # Corrected f-string
    
    # SSL patch for arXiv is removed as fetch_arxiv_papers now uses aiohttp
    
    arxiv_metadata_list: List[Dict[str, Any]] = []
    if arxiv_query and max_results > 0:
        try:
            arxiv_metadata_list = await fetch_arxiv_papers(arxiv_query, max_results) # Changed to await
        except Exception as arxiv_e:
            logger.error(f"arXiv fetching failed: {arxiv_e}", exc_info=True)
            status_messages.append(f"Warning: arXiv fetching failed ({arxiv_e}).")
            arxiv_metadata_list = [] # Ensure it's a list on error
    else:
        arxiv_metadata_list = []

    web_metadata_list: List[Dict[str, Any]] = [] # Ensure type
    if target_urls:
        try:
            # crawl_and_fetch_web_articles is already async
            web_metadata_list = await crawl_and_fetch_web_articles(target_urls)
        except Exception as web_e:
             logger.error(f"Web crawling failed: {web_e}", exc_info=True)
             status_messages.append(f"Warning: Web crawling failed ({web_e}).")
             web_metadata_list = [] # Ensure it's a list on error

        logger.info(f"Web fetcher finished, got {len(web_metadata_list)} results.")
    else:
        web_metadata_list = []

    original_documents = arxiv_metadata_list + web_metadata_list
    num_docs_fetched = len(original_documents)
    if num_docs_fetched == 0:
        # Ensure a string is returned if no data is fetched.
        if not status_messages: # If no specific errors were appended, provide a general message.
            status_messages.append("No data fetched from any source.")
        return "Error: " + " ".join(status_messages)
        
    logger.info(f"Fetched {num_docs_fetched} total documents.")

    # Chunking Documents
    logger.info("Chunking documents...")
    all_chunk_metadata = []
    for doc_index, doc_meta in enumerate(original_documents):
        original_content = doc_meta.get('content', '')
        original_url = doc_meta.get('url', f'doc_{doc_index}')
        if not original_content or not original_content.strip(): continue
        try:
            text_chunks = chunk_text_by_sentences(original_content)
        except Exception as chunk_e: # Catch potential errors from chunking
             logger.error(f"Failed to chunk document {original_url}: {chunk_e}", exc_info=True)
             continue # Skip this document if chunking fails

        if not text_chunks: continue
        for chunk_index, chunk_text in enumerate(text_chunks):
            all_chunk_metadata.append({
                "chunk_id": f"{original_url}_chunk_{chunk_index}", "chunk_text": chunk_text,
                "original_url": original_url, "original_title": doc_meta.get('title', 'Untitled'),
                "source": doc_meta.get('source', 'unknown'), "authors": doc_meta.get('authors', []),
                "published": doc_meta.get('published', None),
                "arxiv_entry_id": doc_meta.get('entry_id') if doc_meta.get('source') == 'arxiv' else None,
                "entry_id": doc_meta.get('entry_id') if doc_meta.get('source') == 'arxiv' else f"{original_url}_chunk_{chunk_index}"
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
    return "".join(lines)

# --- Recommendation Logic ---
# Adjusted return type hint to match implementation
async def run_recommendation(query: str, num_final_results: int, general_mode: bool, concise_mode: bool) -> Tuple[Generator[str, None, None], List[str], List[Dict[str, Any]]]: # Made async
    """
    Handles recommendation: retrieves chunks, prepares context, calls LLM stream.
    Returns a tuple: (response_generator, formatted_source_list, raw_context_chunks_list)
    """
    mode_name = "Hybrid" if general_mode else "Strict"
    style_name = "Concise" if concise_mode else "Detailed"
    logger.info(f"Running recommendation ({mode_name} RAG, {style_name} Prompt) for query: '{query[:100]}...'")

    response_generator: Generator[str, None, None] # Type hint for generator
    formatted_source_list: List[str] = []
    raw_context_chunks_list: List[Dict[str, Any]] = [] # Still needed for return signature
    hybrid_results: List[Tuple[Dict, float]] = []
    captured_exception = None # Variable to hold exception if needed

    try:
        # 1. Load Components
        if loaded_recommender is None or loaded_metadata is None:
             logger.info("Components not loaded, attempting load...")
             load_components() # synchronous call
        if loaded_recommender is None or loaded_metadata is None:
             raise RuntimeError("Core components failed to load. Run 'fetch' command first.")

        # 2. Retrieval Step
        logger.info("Retrieving relevant document chunks...")
        num_candidates = max(config.RAG_NUM_DOCS + 5, num_final_results + 5)
        # Ensure num_candidates doesn't exceed available metadata
        num_candidates = min(num_candidates, len(loaded_metadata))
        if num_candidates > 0:
            rec_params = RecommendationParams(
                semantic_candidates=config.SEMANTIC_CANDIDATES,
                keyword_candidates=config.KEYWORD_CANDIDATES,
                fusion_k=config.RANK_FUSION_K,
                top_n_final=num_candidates
            )
            hybrid_results = loaded_recommender.recommend(
                query=query,
                resource_metadata=loaded_metadata,
                resource_embeddings=loaded_embeddings,
                bm25_index=loaded_bm25_index,
                params=rec_params
            )
            logger.info(f"Retrieved {len(hybrid_results)} candidate chunks.")
        else: logger.warning("Skipping retrieval (no metadata available or zero candidates needed).")

        # 3. Prepare Context, Formatted Sources, AND Raw Chunks
        context_string = "No relevant document chunks were found in the local data."
        reference_map = {}
        if hybrid_results:
            top_chunks_for_rag = hybrid_results[:config.RAG_NUM_DOCS]
            # Store raw chunk dictionaries (still needed for the return tuple)
            raw_context_chunks_list = [chunk_meta for chunk_meta, _ in top_chunks_for_rag]
            context_texts = []
            for i, chunk_meta in enumerate(raw_context_chunks_list):
                 chunk_num = i + 1
                 title = chunk_meta.get('original_title', 'N/A')
                 url = chunk_meta.get('original_url', '#')
                 snippet = (chunk_meta.get('chunk_text', '') or '')[:config.MAX_CONTEXT_LENGTH_PER_DOC]
                 # Adjusted context format slightly for clarity
                 context_texts.append(f"Source [{chunk_num}]:Title: {title} URL: {url} Content Snippet: {snippet}")
                 reference_map[chunk_num] = f"{title} (URL: {url})"
            context_string = "---".join(context_texts)
            # Create the formatted list for display
            formatted_source_list = [f"[{num}] {details}" for num, details in reference_map.items()]
            logger.info(f"Prepared context from {len(raw_context_chunks_list)} chunks.")
        else:
            logger.info("No relevant document chunks found to provide as context.")
            formatted_source_list = ["No relevant document chunks found."]

        # 4. Prompt Selection (Simplified without explicit citation instructions)
        final_prompt: str = ""
        prompt_template_base = """[INST] {system_message}

Provided Document Context:
{context}

User Question: {query} [/INST]
Answer:"""

        if general_mode:
            if concise_mode:
                system_message = ( # Hybrid + Concise (No Citation)
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
                system_message = ( # Hybrid + Detailed (No Citation)
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
                system_message = ( # Strict + Concise (No Citation)
                     "You are an AI assistant expert at analyzing technical documents."
                     "Provide a concise summary answering the user's question using **only** information found in the 'Provided Document Context'. **Do not use any outside knowledge.**"
                     "**Format your entire response using standard Markdown.**"
                     "Use the following structure:"
                     "## Summary [Provide a brief overview answering the main question based *only* on the context.]"
                     "## Key Findings [List the most important findings or points using bullet points (* or -). Extract information directly from the context.]"
                     "## Conclusion [Summarize the main takeaways based *only* on the context. State if the provided context is insufficient to fully answer.]"
                )
            else:
                system_message = ( # Strict + Detailed (No Citation)
                     "You are an AI assistant expert at analyzing technical documents."
                     "Provide a detailed answer using **only** information found in the 'Provided Document Context'. **Do not use any outside knowledge.**"
                     "**Structure your response clearly using standard Markdown:**"
                     "**## Overview:** [Start with a concise paragraph summarizing the main answer based *only* on the context.]"
                     "**## Detailed Analysis:** [Use ### Sub-Headings for distinct topics found *only* in the documents. Under each sub-heading, explain the topic thoroughly using multiple sentences or bullet points (* or -). Synthesize information across different context sources where applicable, but stick strictly to the provided text.]"
                     "**## Conclusion:** [Provide a concluding paragraph summarizing the key points based *only* on the context. Explicitly state if the information in the context is limited or insufficient.]"
                )

        final_prompt = prompt_template_base.format(system_message=system_message, context=context_string, query=query)

        # --- REMOVED Redundant Log Line ---
        # logger.info(f"Sending STREAMING request to LLM ({config.LLM_PROVIDER}, {config.LLM_MODEL_ID})...")
        logger.info("Preparing to call LLM for response generation...") # More general log
        response_generator = get_llm_response_stream(final_prompt)

    except RuntimeError as e:
        logger.error(f"Recommendation failed due to component loading error: {e}", exc_info=False)
        captured_exception = e # Capture exception
        # --- FIXED: Pass captured exception to error_generator ---
        def error_generator(err): yield f"Error: {err}. Cannot run recommendation."
        response_generator = error_generator(captured_exception)
        formatted_source_list = []
        raw_context_chunks_list = [] # Ensure it's empty on error
    except Exception as e:
        logger.error(f"Unexpected error during recommendation setup: {e}", exc_info=True)
        captured_exception = e # Capture exception
        # --- FIXED: Pass captured exception to error_generator ---
        def error_generator(err): yield f"An unexpected error occurred during setup: {err}"
        response_generator = error_generator(captured_exception)
        formatted_source_list = []
        raw_context_chunks_list = [] # Ensure it's empty on error

    # Return the generator, formatted sources, and raw chunks
    return response_generator, formatted_source_list, raw_context_chunks_list


# --- arXiv Search Logic ---
def run_arxiv_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    """Performs a direct arXiv search."""
    logger.info(f"Searching arXiv directly for: '{query}' (Top {num_results})")
    results_list: List[Dict[str, Any]] = []
    
    # Try to import and apply our SSL patch
    try:
        # Path to arxiv_ssl_patch module
        sys.path.insert(0, os.path.join(script_dir))
        from arxiv_ssl_patch import arxiv_ssl_patch, with_arxiv_retry
        arxiv_ssl_patch()
        
        # Create client with retry decorator
        @with_arxiv_retry(max_attempts=5, min_wait=2.0, max_wait=30.0)
        def create_arxiv_client():
            return arxiv.Client(page_size=min(num_results, 100), delay_seconds=1.0, num_retries=5)
        
        client = create_arxiv_client()
    except ImportError:
        # Fall back to normal client if patch isn't available
        logger.warning("SSL patch module not found; using standard client (vulnerable to SSL errors)")
        client = arxiv.Client(page_size=min(num_results, 100), delay_seconds=1.0, num_retries=3)
    
    try:
        search = arxiv.Search(query=query, max_results=num_results, sort_by=arxiv.SortCriterion.Relevance)
        for result in client.results(search):
            results_list.append({
                "title": result.title or "N/A",
                "authors": [str(a) for a in result.authors],
                "published": result.published.strftime('%Y-%m-%d') if result.published else "N/A",
                "pdf_url": result.pdf_url or 'N/A',
                "summary": (result.summary or '').replace('', ' ').strip() # Clean summary
            })
        logger.info(f"arXiv search found {len(results_list)} results.")
    except ssl.SSLError as ssl_err:
        logger.error(f"SSL error during arXiv search: {ssl_err}", exc_info=True)
        logger.info("SSL issue detected. Consider using the arxiv_ssl_patch.py script directly.")
    except ConnectionError as conn_err:
        logger.error(f"Connection error during arXiv search: {conn_err}", exc_info=True)
    except Exception as e:
        logger.error(f"arXiv search failed: {e}", exc_info=True)
    return results_list


# --- Main CLI Execution ---
async def main(): # Make main async
    parser = argparse.ArgumentParser(description='''CLI for Hybrid Search RAG.
                                     Example: python scripts/cli.py "What is RAG?" --data_path ./data_hybrid --sources arxiv,web --recommend --num_results 5 --num_final_results 3''')
    parser.add_argument("query", type=str, help="The query to search for.")
    parser.add_argument("--data_path", type=str, default=config.DATA_DIR, help="Path to the data directory.")
    parser.add_argument("--sources", type=str, default="all", help="Comma-separated sources to search (e.g., arxiv,web,local). Default is all.")
    parser.add_argument("--recommend", action="store_true", help="Enable LLM-based recommendation.")
    parser.add_argument("-n", "--num_results", type=int, default=config.TOP_N_RESULTS, help="Number of search results to display.")
    parser.add_argument("-f", "--num_final_results", type=int, default=config.RAG_NUM_DOCS, help="Number of final results for LLM processing.")
    parser.add_argument("--stream", action="store_true", help="Stream the LLM response if recommendation is enabled.")
    parser.add_argument("--no_cache", action="store_true", help="Force refetch and reprocess data.")
    parser.add_argument("--fetch_arxiv", action="store_true", help="Fetch new papers from arXiv based on the query.")
    parser.add_argument("--arxiv_query", type=str, default=config.DEFAULT_ARXIV_QUERY, help="Custom query for fetching arXiv papers.")
    parser.add_argument("--num_arxiv", type=int, default=config.MAX_ARXIV_RESULTS, help="Max number of arXiv papers to fetch.")
    parser.add_argument("--crawl_web", action="store_true", help="Crawl web pages from TARGET_WEB_URLS in config.")
    parser.add_argument("--rebuild_index", action='store_true', help="Force rebuild of BM25 index and embeddings from existing metadata (if fetch is not run).")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")
    parser.add_argument("--stream", action='store_true', default=True, help="Stream LLM responses. Default: True.") # Default to True
    parser.add_argument("--no-stream", dest='stream', action='store_false', help="Disable streaming LLM responses.")


    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled.")

    try:
        check_nltk_data() # Initial check, especially if not fetching
    except SystemExit:
        print("Exiting: NLTK data setup failed.", file=sys.stderr)
        sys.exit(1)


    if args.fetch:
        print("Fetching and processing data...")
        # setup_data_and_fetch is now async, so we need to run it in an event loop
        fetch_status = await setup_data_and_fetch(args) # await async call
        print(fetch_status)
        if fetch_status.startswith("Error:"):
            sys.exit(1) # Exit if fetching failed critically
        # Components are reloaded within setup_data_and_fetch on success
        print("Data fetching and processing complete. You can now run queries.")
        # Decide if to exit or proceed to query/interactive mode
        if not args.query: # If no query was provided with fetch, exit.
            print("Exiting after fetch. Run again with a query or in interactive mode.")
            sys.exit(0)
        # If a query was provided with --fetch, proceed to execute it after loading
        try:
            # load_components is already called by setup_data_and_fetch on success.
            # If fetch_status did not start with "Error:", components should be loaded.
            # However, if there was a partial success, an explicit load might be needed.
            # For safety, ensure components are loaded if recommender is still None.
            if loaded_recommender is None:
                 logger.info("Ensuring components are loaded after fetch for query execution...")
                 load_components() # synchronous call
        except RuntimeError as e:
            print(f"Error loading components after fetch: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.rebuild_index:
        print("Rebuilding index and embeddings from existing data...")
        try:
            load_components(force_reload=False) # Load existing metadata first
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
                    try: # Embedding
                        embeddings = embedder.encode(all_chunk_texts, task_type="RETRIEVAL_DOCUMENT")
                        if embeddings is not None and isinstance(embeddings, np.ndarray) and embeddings.shape[0] == len(all_chunk_texts):
                            logger.info(f"Embeddings regenerated (Shape: {embeddings.shape}).")
                            embed_success = True
                        else: logger.error("Embedding regeneration failed or returned unexpected result.")
                    except Exception as e:
                        logger.error(f"Embedding regeneration failed: {e}", exc_info=True)

                    try: # BM25 Indexing
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

                    data_manager.save_all_data(loaded_metadata, embeddings, bm25_index) # Save potentially new embeddings/index
                    load_components(force_reload=True) # Reload all components
                    print(f"Rebuild complete. Embeddings updated: {embed_success}. BM25 index updated: {bm25_success}.")
            else:
                print("No metadata loaded. Cannot rebuild. Run --fetch first.")
        except RuntimeError as e:
            print(f"Error during rebuild: {e}", file=sys.stderr)
            sys.exit(1)
        if not args.query: # If no query was provided with rebuild, exit.
            print("Exiting after rebuild. Run again with a query or in interactive mode.")
            sys.exit(0)


    # --- Ensure components are loaded before query processing or interactive mode ---
    if loaded_recommender is None: # Check if components were loaded by fetch/rebuild
        try:
            logger.info("Loading components for query/interactive mode...")
            load_components() # synchronous call
        except RuntimeError as e:
            print(f"Critical Error: Failed to load necessary components: {e}", file=sys.stderr)
            print("Try running with --fetch to (re)generate data or check data file paths.", file=sys.stderr)
            sys.exit(1)
        if loaded_recommender is None: # Still None after attempt
             print("Critical Error: Recommender could not be initialized even after loading attempt.", file=sys.stderr)
             sys.exit(1)
    # --- End Component Loading Check ---


    general_mode = args.mode == 'general'
    concise_mode = args.concise

    if args.query:
        # Single query mode
        print(f"Processing query: \'{args.query}\' (Mode: {args.mode}, Concise: {args.concise}, Stream: {args.stream})")
        response_gen, sources, _ = await run_recommendation(args.query, args.num_final_results, general_mode, concise_mode) # await async call
        
        full_response = []
        if args.stream:
            print("\nLLM Response:")
            try:
                for chunk in response_gen:
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
                print() # Newline after streaming
            except Exception as e:
                print(f"\nError during streaming: {e}")
                logger.error(f"Error during streaming LLM response: {e}", exc_info=True)
                # Fallback or alternative display can be added here
        else: # Not streaming
            print("\nLLM Response (non-streamed):")
            try:
                # Consume the generator to get the full response
                full_response_content = "".join(list(response_gen))
                print(full_response_content)
                full_response.append(full_response_content) # Store for consistency if needed later
            except Exception as e:
                print(f"\nError processing non-streamed response: {e}")
                logger.error(f"Error processing non-streamed LLM response: {e}", exc_info=True)
            # Ensure there is a finally or except clause for the try statement at the end of the non-streaming block
            # This was previously missing, causing "Try statement must have at least one except or finally clause"
            # Adding a pass for now, assuming errors are caught by the outer try/except in main loop or specific error handling above.
            # If specific cleanup is needed, it should go in a `finally` block.
            finally:
                pass 

        print("\nSources:")
        for source in sources:
            print(f"- {source}")

    else:
        # Interactive mode
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() == 'exit':
                    print("Exiting interactive mode.")
                    break

                response_gen, sources, _ = await run_recommendation(user_input, args.num_final_results, general_mode, concise_mode) # await async call
                
                print("\nLLM Response:")
                try:
                    for chunk in response_gen:
                        print(chunk, end="", flush=True)
                except Exception as e:
                    print(f"\nError during streaming: {e}")
                    logger.error(f"Error during streaming LLM response in interactive mode: {e}", exc_info=True)

                print("\nSources:")
                for source in sources:
                    print(f"- {source}")

            except Exception as e:
                print(f"Error processing request: {e}")
                logger.error(f"Error processing request in interactive mode: {e}", exc_info=True)
                # Optionally, provide fallback or recovery actions here
