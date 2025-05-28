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
# Flag to ensure check/download runs only once per script execution if successful.
# If it fails, it might be re-attempted.
_nltk_data_checked_cli_status: Optional[bool] = None


def check_nltk_data() -> bool:
    """
    Checks for required NLTK data packages ('punkt', 'stopwords').
    Attempts to download them if missing to a project-local directory.
    For 'punkt', it specifically tests nltk.sent_tokenize.
    Returns True if all packages are verified and usable, False otherwise.
    """
    global _nltk_data_checked_cli_status
    if _nltk_data_checked_cli_status is True:
        return True

    logger.info("Performing NLTK data check (CLI version)...")

    # --- NLTK Data Path Setup ---
    # project_root is defined globally in this script
    nltk_data_dir = os.path.join(project_root, "nltk_data_ra")
    try:
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
            logger.info(f"Created NLTK data directory: {nltk_data_dir}")
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
            logger.info(f"Prepended {nltk_data_dir} to nltk.data.path.")
        logger.info(f"Current nltk.data.path: {nltk.data.path}")
    except Exception as e_path:
        logger.error(f"Failed to setup NLTK data directory {nltk_data_dir}: {e_path}", exc_info=True)
        # If path setup fails, it's risky to proceed with downloads to unknown locations.
        _nltk_data_checked_cli_status = False
        return False
    # --- End NLTK Data Path Setup ---

    required_nltk_packages = ["punkt", "stopwords"]
    all_packages_ok = True

    for package_name in required_nltk_packages:
        package_verified = False
        try:
            if package_name == "punkt":
                # Test 1: Find the main English pickle file
                nltk.data.find("tokenizers/punkt/PY3/english.pickle")
                logger.info(f"NLTK package '{package_name}' (english.pickle) found.")
                # Test 2: Perform actual sentence tokenization
                nltk.sent_tokenize("This is a test sentence. This ensures punkt is fully functional.")
                logger.info(f"NLTK package '{package_name}' successfully tested with sent_tokenize.")
                package_verified = True
            elif package_name == "stopwords":
                nltk.data.find("corpora/stopwords/english") # Check for English stopwords list
                logger.info(f"NLTK package '{package_name}' (english stopwords) found.")
                # Optionally, load them: from nltk.corpus import stopwords; stopwords.words('english')
                package_verified = True
            
            if not package_verified: # Should not happen if above checks pass without LookupError
                 raise LookupError(f"Initial check for {package_name} passed but verification logic failed.")        except LookupError as e:
            logger.warning(f"NLTK package '{package_name}' not found or initial test failed. Attempting download to {nltk_data_dir}...")
            try:
                # Try downloading punkt_tab first if the error mentions it
                if "punkt_tab" in str(e):
                    logger.info(f"LookupError mentions 'punkt_tab', attempting to download 'punkt_tab' first...")
                    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
                    logger.info(f"NLTK package 'punkt_tab' downloaded to {nltk_data_dir}.")
                
                # Always try the original package too
                nltk.download(package_name, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"NLTK package '{package_name}' downloaded to {nltk_data_dir}.")
                # Re-verify after download
                if package_name == "punkt":
                    nltk.data.find("tokenizers/punkt/PY3/english.pickle") # Re-check pickle
                    nltk.sent_tokenize("This is a test sentence after download. For NLTK punkt.") # Re-test tokenize
                    logger.info(f"NLTK package '{package_name}' successfully re-tested after download.")
                elif package_name == "stopwords":
                    nltk.data.find("corpora/stopwords/english")
                    logger.info(f"NLTK package '{package_name}' (english stopwords) verified after download.")
                package_verified = True
            except Exception as e:
                logger.error(f"Failed to download or verify NLTK package '{package_name}' after download attempt: {e}", exc_info=True)
                all_packages_ok = False
        except Exception as e_test: # Catch other errors from tests (e.g., sent_tokenize)
            logger.error(f"NLTK package '{package_name}' test failed: {e_test}", exc_info=True)
            all_packages_ok = False
        
        if not package_verified and all_packages_ok: # If a package wasn't verified but no error set all_packages_ok to False
            all_packages_ok = False


    if not all_packages_ok:
        logger.critical("One or more NLTK data packages are missing or failed verification. Chunking and other NLP tasks may fail.")
        _nltk_data_checked_cli_status = False # Mark as failed
    else:
        logger.info("All required NLTK data packages verified and tested successfully (CLI version).")
        _nltk_data_checked_cli_status = True # Mark as successful

    return _nltk_data_checked_cli_status


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
def load_components(force_reload: bool = False) -> Dict[str, Any]: # Made sync - no async operations needed
    global loaded_metadata, loaded_embeddings, loaded_bm25_index, loaded_recommender
    # Initialize a dictionary to hold the components
    components_to_return: Dict[str, Any] = {
        "data_manager": None,
        "recommender": None,
        "llm_interface": None, # Assuming llm_interface is also a component to be returned
        "status_message": "Initialization pending.", # Added status message
        "nltk_manager": None # Added NltkManager
    }

    if force_reload or loaded_metadata is None or loaded_recommender is None:
        logger.info(f"{'Forcing reload' if force_reload else 'Loading core components'}...")
        components_to_return["status_message"] = "Loading core components..."
        
        data_manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
        components_to_return["data_manager"] = data_manager

        # Load data and cast bm25_index to its expected type
        logger.info(f"Attempting to load data from DataManager. DATA_DIR: {config.DATA_DIR}, METADATA_FILE: {config.METADATA_FILE}")
        meta_temp, embeddings_temp, bm25_index_object = data_manager.load_all_data()
        loaded_metadata = meta_temp
        loaded_embeddings = embeddings_temp
        loaded_bm25_index = cast(Optional[BM25Okapi], bm25_index_object)

        if loaded_metadata is None:
            error_msg = f"Metadata loading failed from {data_manager.metadata_path}. Cannot initialize recommender. Knowledge base might be empty or inaccessible."
            logger.error(error_msg)
            components_to_return["status_message"] = error_msg
            return components_to_return
        else:
            logger.info(f"Successfully loaded {len(loaded_metadata)} metadata items.")
            components_to_return["status_message"] = f"Loaded {len(loaded_metadata)} metadata items."

        try:
            # Ensure NLTK data needed by Recommender/NltkManager is checked/loaded
            logger.info("Checking NLTK data for component loading...")
            if not check_nltk_data(): # Now returns bool
                error_msg = "NLTK data check/download failed. Recommender cannot be initialized."
                logger.critical(error_msg)
                components_to_return["status_message"] = error_msg
                return components_to_return
            logger.info("NLTK data check successful.")
            components_to_return["status_message"] += " NLTK data OK."

            # Initialize NltkManager first
            nltk_manager = NltkManager()
            components_to_return["nltk_manager"] = nltk_manager
            logger.info("NltkManager initialized.")

            embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
            # HybridRecommender initializes its own NltkManager
            loaded_recommender = HybridRecommender(embed_model=embedder)
            components_to_return["recommender"] = loaded_recommender
            logger.info("Recommender initialized.")
            components_to_return["status_message"] += " Recommender initialized."
            
            # Placeholder for LLM Interface if it needs to be part of loaded components
            # For now, assuming llm_interface is globally accessible or not part of this specific dict
            # If it is, it should be initialized and added here:
            # from hybrid_search_rag.llm_services.llm_interface import SomeLLMInterfaceClass
            # components_to_return["llm_interface"] = SomeLLMInterfaceClass()
            # For the purpose of app.py checking, let's ensure it's present, even if None for now
            if "llm_interface" not in components_to_return or components_to_return["llm_interface"] is None:
                 # This part depends on how llm_interface is structured.
                 # If it's a module with functions like get_llm_response, we might not need to "load" an instance.
                 # For now, let's assume app.py checks for its presence in the returned dict.
                 # A simple way is to assign the module itself if app.py expects an object with methods.
                 try:
                     from hybrid_search_rag.llm_services import llm_interface
                     components_to_return["llm_interface"] = llm_interface # Assign the module
                     logger.info("LLM Interface module assigned to components.")
                 except ImportError as e:
                     logger.error(f"Failed to import llm_interface module for components_to_return: {e}")
                     # components_to_return["llm_interface"] will remain None

        except SystemExit: # Catch exit from check_nltk_data
             logger.critical("NLTK data check failed during component loading. Cannot initialize recommender.")
             # Return partially filled components
             return components_to_return
             # raise RuntimeError("Failed to acquire NLTK data during component loading.") # Original behavior
        except Exception as e:
            logger.error(f"Recommender initialization failed: {e}", exc_info=True)
            loaded_recommender = None
            components_to_return["recommender"] = None
            components_to_return["status_message"] = f"Recommender init failed: {e}"
            return components_to_return

        if loaded_embeddings is None: 
            logger.warning("Embeddings file not found or invalid; semantic search disabled.")
            components_to_return["status_message"] += " Embeddings missing."
        if loaded_bm25_index is None: 
            logger.warning("BM25 index file not found or invalid; keyword search disabled.")
            components_to_return["status_message"] += " BM25 index missing."
        if loaded_embeddings is None and loaded_bm25_index is None and loaded_metadata is not None:
            warning_msg = "Both embeddings and BM25 index are missing. Search functionality severely limited."
            logger.error(warning_msg)
            # If metadata IS present, this is a partial load, not a complete failure to find data.
            components_to_return["status_message"] += f" {warning_msg}"
        elif loaded_metadata is None: # This case should be caught earlier by the metadata check
            components_to_return["status_message"] = "Critical: Metadata is missing. All components failed to load."

    else:
        logger.info("Core components already loaded, returning existing instances.")
        components_to_return["data_manager"] = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
        components_to_return["recommender"] = loaded_recommender
        components_to_return["nltk_manager"] = NltkManager() # Assuming NltkManager is stateless or re-init is fine
        try:
            from hybrid_search_rag.llm_services import llm_interface
            components_to_return["llm_interface"] = llm_interface
            components_to_return["status_message"] = "Components already loaded."
        except ImportError as e:
            logger.error(f"Failed to import llm_interface module for already loaded components: {e}")
            components_to_return["llm_interface"] = None
            components_to_return["status_message"] = "Components loaded, but LLM interface import failed."

    # Final check on recommender status for the message
    if components_to_return.get("recommender") is None and "Recommender init failed" not in components_to_return.get("status_message", "") and "Metadata loading failed" not in components_to_return.get("status_message", "") :
        components_to_return["status_message"] = components_to_return.get("status_message", "") + " Recommender is None (check logs)."
    elif components_to_return.get("recommender") is not None and "Recommender initialized" not in components_to_return.get("status_message", "") and "Components already loaded" not in components_to_return.get("status_message", ""):
        components_to_return["status_message"] = components_to_return.get("status_message", "") + " Recommender loaded."

    logger.info(f"load_components returning with status: {components_to_return.get('status_message')}")
    return components_to_return

# --- Data Fetching Logic ---
async def setup_data_and_fetch(args: argparse.Namespace) -> str:
    """Handles the 'fetch' command: gets documents, chunks, embeds, indexes, and saves."""
    logger.info("Starting data fetch and processing...")
    status_messages = []

    # --- Ensure NLTK data is available before proceeding ---
    try:
        # Call the more robust check_nltk_data
        if not check_nltk_data():
            # This message will be returned to the UI if fetch is triggered from there
            return "Error: Critical NLTK data (e.g., 'punkt' for sentence tokenization) is missing or failed verification. Cannot proceed with data fetching. Please check logs."
    except Exception as e: # Catch any unexpected error from check_nltk_data
        logger.critical(f"An unexpected error occurred during NLTK data check: {e}", exc_info=True)
        return f"Error: Unexpected issue during NLTK data check: {e}. Cannot proceed."
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
        use_playwright_pdf = getattr(args, 'use_playwright_for_pdfs', True) # Default to True
        fetch_timeout = getattr(args, 'fetch_timeout_seconds', 30) # Default to 30 seconds

        fetcher = ResourceFetcher(
            timeout=fetch_timeout,
            use_playwright_for_pdfs=use_playwright_pdf
            # cache_dir and rate_limiter can be added if needed, using defaults for now
        )
        # Pass the verbose/debug flag to fetch_arxiv_papers and other relevant calls if they accept it.
        # The verbose argument for ResourceFetcher itself is not directly in its __init__ based on the provided snippet,
        # but its logger can be configured. For now, we ensure args.debug is safely accessed for other functions.
        
        processed_details_all = {"total_new_items": 0, "errors": []}

        logger.info(f"arXiv PDF metadata fetching enabled. Verbose/Debug: {getattr(args, 'debug', False)}")

        if arxiv_query and max_results > 0:
            try:
                arxiv_metadata_list_with_potential_pdfs = await fetch_arxiv_papers(
                    query=arxiv_query,
                    max_results=max_results,
                    verbose=getattr(args, 'debug', False) # Pass debug flag for verbose logging
                )
                processed_arxiv_docs = []
                for paper_meta in arxiv_metadata_list_with_potential_pdfs:
                    if paper_meta.get('pdf_url'):
                        logger.info(f"Attempting to fetch PDF content for: {paper_meta.get('title')} from {paper_meta.get('pdf_url')}")
                        # Use ResourceFetcher.fetch_document for PDF
                        # fetch_document is async, so await it.
                        # It handles the sync playwright call in a thread.
                        fetched_doc_data = await fetcher.fetch_document(
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
        
        # Check if we need to proceed with general knowledge only (empty knowledge base)
        kb_is_empty = loaded_metadata is None or len(loaded_metadata) == 0
        
        if kb_is_empty:
            # Don't raise an error for empty knowledge base in general mode
            if general_mode:
                logger.warning("Knowledge base is empty, proceeding with general knowledge only.")
            else:
                # Only raise error if not in general mode
                raise RuntimeError("Core components failed to load and general_mode is False. Run 'fetch' command first.")

        # Check if knowledge base exists and has content
        kb_is_empty = loaded_metadata is None or len(loaded_metadata) == 0
        
        if kb_is_empty:
            logger.warning("Knowledge base is empty, skipping retrieval step.")
            hybrid_results = []  # Empty results list since we can't perform retrieval
        else:
            logger.info("Retrieving relevant document chunks...")
            num_candidates = max(config.RAG_NUM_DOCS + 5, num_final_results + 5)
            metadata_length = len(loaded_metadata) if loaded_metadata is not None else 0
            num_candidates = min(num_candidates, metadata_length)
            if num_candidates > 0:
                rec_params = RecommendationParams(
                    semantic_candidates=config.SEMANTIC_CANDIDATES,
                    keyword_candidates=config.KEYWORD_CANDIDATES,
                    fusion_k=config.RANK_FUSION_K,
                    top_n_final=num_candidates
                )
                # Ensure loaded_recommender is not None and loaded_metadata is not None before calling recommend
                if loaded_recommender and loaded_metadata is not None:
                    hybrid_results = loaded_recommender.recommend(
                        query=query,
                        resource_metadata=loaded_metadata,
                        resource_embeddings=loaded_embeddings,
                        bm25_index=loaded_bm25_index,
                        params=rec_params
                    )
                else: # Should be caught by earlier check, but as a safeguard
                    if general_mode:
                        logger.warning("Recommender or metadata is None but general_mode is True, proceeding without retrieval.")
                        hybrid_results = []
                    else:
                        raise RuntimeError("Recommender or metadata is None, cannot proceed with recommendation.")
                logger.info(f"Retrieved {len(hybrid_results)} candidate chunks.")
            else: logger.warning("Skipping retrieval (no metadata available or zero candidates needed).")

        # Determine if we have an empty knowledge base scenario
        kb_is_empty = loaded_metadata is None or len(loaded_metadata) == 0
        
        # Set a contextual message for empty knowledge base or no relevant chunks
        if kb_is_empty:
            context_string = "The knowledge base is empty. No document chunks are available for context."
            logger.info("Knowledge base is empty - proceeding with empty context.")
        else:
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
            if kb_is_empty:
                logger.info("Knowledge base is empty - no context available.")
                formatted_source_list = ["Knowledge base is empty - no documents available."]
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

