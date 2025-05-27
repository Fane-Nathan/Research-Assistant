# -*- coding: utf-8 -*-
"""
Configuration settings for the Hybrid Search RAG project.

Organizes settings into logical sections for better readability and management.
Handles environment variables for sensitive information like API keys.
Includes options for LLM fallback across multiple models within providers,
and across different providers (e.g., Google -> Groq).
"""

import os
import logging
import sys # Import sys for stderr printing
from typing import List, Optional, Final # Added Final for constants

# --- Basic Logging Setup (Configure early) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)-8s - [%(name)s - %(funcName)s:%(lineno)d] - %(message)s', # Added more details to format
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True # force=True can be useful if other libs try to configure logging, but use with awareness
)
logger = logging.getLogger(__name__)

# --- Project Root Determination ---
_project_root_value: str
try:
    _project_root_value = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Project root determined: {_project_root_value}")
except Exception as e:
    logger.error(f"Failed to determine project root: {e}", exc_info=True)
    _project_root_value = os.getcwd() # Fallback
    logger.warning(f"Falling back to current working directory as project root: {_project_root_value}")

PROJECT_ROOT: Final[str] = _project_root_value


# --- Environment Variable Loading (.env) ---
DOTENV_PATH: Final[str] = os.path.join(PROJECT_ROOT, '.env')
try:
    from dotenv import load_dotenv
    if os.path.exists(DOTENV_PATH):
        load_dotenv(dotenv_path=DOTENV_PATH)
        logger.info(f"Loaded environment variables from: {DOTENV_PATH}")
    else:
        logger.warning(f".env file not found at {DOTENV_PATH}. API keys might be missing.")
except ImportError:
    logger.warning("`python-dotenv` package not found. Cannot load .env file.")
except Exception as e:
    logger.error(f"Error loading .env file from {DOTENV_PATH}: {e}", exc_info=True)


# ==============================================================================
# --- Core Paths ---
# ==============================================================================
# --- Data Storage Configuration ---
# Defines where data like metadata, embeddings, and BM25 indexes are stored.
# DATA_DIR: Final[str] = os.path.join(PROJECT_ROOT, "data_store", "corpus_data") # Original
DATA_DIR: Final[str] = os.path.join(PROJECT_ROOT, "data_hybrid") # Changed to data_hybrid
METADATA_FILE: Final[str] = "combined_metadata.json"
EMBEDDINGS_FILE: Final[str] = "combined_embeddings.npy"
BM25_INDEX_FILE: Final[str] = "bm25_index.pkl"


# ==============================================================================
# --- API Keys & Cloud Configuration ---
# ==============================================================================
# Load API keys from environment variables (set in .env file).
GOOGLE_API_KEY: Final[Optional[str]] = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY: Final[Optional[str]] = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY: Final[Optional[str]] = os.getenv("DEEPSEEK_API_KEY") # Added DeepSeek API Key
# Add other keys if needed (e.g., HF_API_TOKEN)


# ==============================================================================
# --- LLM Configuration ---
# ==============================================================================
# --- Provider Order ---
# Define the order in which to try providers. The first provider is primary.
LLM_PROVIDER_ORDER: Final[List[str]] = ["google", "groq", "deepseek"] # Example: Try Google first, then Groq, then DeepSeek

# --- Models per Provider ---
# Define lists of model IDs to try *within* each provider, in order of preference.
# The interface will iterate through these models for the first provider in
# LLM_PROVIDER_ORDER, then move to the models for the second provider, etc.
LLM_GOOGLE_MODELS: Final[List[str]] = [
    "gemini-2.0-flash",                # Older alias
    # "gemini-2.5-flash-preview-05-20",
    # "gemini-2.5-pro-preview-03-25",   # Fallback Google choice
]
LLM_GROQ_MODELS: Final[List[str]] = [
    "qwen-qwq-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",    # Primary Groq choice
    "meta-llama/llama-4-scout-17b-16e-instruct",        # Fallback Groq choice
]
LLM_DEEPSEEK_MODELS: Final[List[str]] = [
    "deepseek-chat",  # Replace with actual DeepSeek model ID(s)
    # "deepseek-coder", # Example of another model
]

# --- General LLM Parameters ---
# These parameters will be attempted for all API calls.
LLM_TEMPERATURE: Final[float] = 0.3
LLM_MAX_NEW_TOKENS: Final[int] = 4096
LLM_API_TIMEOUT: Final[int] = 60 # Timeout in seconds (adjust as needed)


# ==============================================================================
# --- Embedding Model Configuration ---
# ==============================================================================
EMBEDDING_MODEL_NAME: Final[str] = 'models/text-embedding-004'
EMBEDDING_DIM: Final[int] = 768
VERTEX_AI_BATCH_LIMIT: Final[int] = 100


# ==============================================================================
# --- Data Source Settings ---
# ==============================================================================
DEFAULT_ARXIV_QUERY: Final[str] = ""
MAX_ARXIV_RESULTS: Final[int] = 100
TARGET_WEB_URLS: Final[List[str]] = [
    # "https://distill.pub/",
]
MAX_PAGES_TO_CRAWL: Final[int] = 200
CRAWL_DELAY_SECONDS: Final[float] = 0.5
ALLOWED_DOMAINS: Final[List[str]] = []
FETCH_TIMEOUT: Final[int] = 45
HEAD_TIMEOUT: Final[int] = 20


# ==============================================================================
# --- Retrieval & Ranking Settings ---
# ==============================================================================
DEFAULT_QUERY: Final[str] = "Explain Retrieval-Augmented Generation (RAG)"
TOP_N_RESULTS: Final[int] = 10
SEMANTIC_CANDIDATES: Final[int] = 150
KEYWORD_CANDIDATES: Final[int] = 150
RANK_FUSION_K: Final[int] = 60


# ==============================================================================
# --- RAG (Retrieval-Augmented Generation) Settings ---
# ==============================================================================
RAG_NUM_DOCS: Final[int] = 15
MAX_CONTEXT_LENGTH_PER_DOC: Final[int] = 2000


# ==============================================================================
# --- Evaluation Script Settings ---
# ==============================================================================
# For scripts like auto_generate_evaluation_dataset.py
MAX_CANDIDATES_PER_QUERY_TO_LABEL_CONFIG: Final[Optional[int]] = 10
DELAY_BETWEEN_LLM_CALLS_SECONDS_CONFIG: Final[float] = 5.0
MAX_LLM_RETRIES_CONFIG: Final[int] = 2
RETRY_ON_EMPTY_LLM_RESPONSE_CONFIG: Final[bool] = False
RETRY_DELAY_SECONDS_SHORT_CONFIG: Final[int] = 5
LONG_RETRY_DELAY_SECONDS_FOR_RATE_LIMIT_CONFIG: Final[int] = 65
LLM_TEMPERATURE_EVAL_CONFIG: Final[float] = 0.1 # Temperature specifically for evaluation LLM calls

# ==============================================================================
# --- arXiv Data Collection Script Settings (arxiv_parallel_dataset_retrieval.py) ---
# ==============================================================================
# These settings are for the scripts/data_collection/arxiv_parallel_dataset_retrieval.py script

# Primary arXiv categories for the initial broad search phase.
ARXIV_PROC_PRIMARY_TARGET_CATEGORIES: Final[List[str]] = [
    'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.RO', 'cs.NE', 'cs.IR', 'cs.HC',
    'cs.ET', 'cs.SE', 'cs.DB', 'cs.DC', 'cs.CR', 'cs.NI', 'cs.MA', 'cs.SY',
    'stat.ML'
]

# Max results to attempt to fetch from arXiv API for each primary category search.
ARXIV_PROC_MAX_RESULTS_PER_PRIMARY_CATEGORY_SEARCH: Final[int] = 5000

# Secondary filter: List of categories to INCLUDE (paper must have AT LEAST ONE matching prefix).
ARXIV_PROC_FILTER_CATEGORIES: Final[List[str]] = sorted(list(set([
    'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.CY',
    'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR',
    'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS',
    'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC',
    'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY',
    'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY',
    'stat.ML', 'stat.CO', 'stat.AP',
    'physics.app-ph', 'physics.ins-det', 'physics.optics', 'physics.comp-ph',
    'physics.data-an', 'physics.flu-dyn', 'physics.acc-ph', 'physics.plasm-ph', 'physics.space-ph',
    'physics.geo-ph', 'physics.med-ph',
    'math.IT', 'math.NA', 'math.OC', 'math.DS',
    'q-bio.CB', 'q-bio.GN', 'q-bio.QM', 'q-bio.BM', 'q-bio.SC', 'q-bio.TO',
    'q-fin.CP', 'q-fin.ST', 'q-fin.TR', 'q-fin.RM',
    'nlin.AO', 'nlin.PS',
    'quant-ph'
])))

# Overall maximum number of *new* papers to process in a single run.
ARXIV_PROC_MAX_PAPERS_TO_PROCESS: Final[Optional[int]] = 100000

# Number of parallel worker threads for downloading and parsing.
ARXIV_PROC_NUM_WORKERS: Final[int] = 12

# Timeout in seconds for downloading each PDF.
ARXIV_PROC_REQUESTS_TIMEOUT: Final[int] = 20

# Base delay in seconds between starting network calls for EACH paper by a worker.
ARXIV_PROC_API_DELAY_PER_ID: Final[float] = 0.5

# Batch size for processing discovered IDs (helps manage memory).
ARXIV_PROC_PROCESSING_BATCH_SIZE: Final[int] = 100000 # Default, can be overridden by CLI

# Tenacity retry parameters for PDF downloads
ARXIV_PROC_RETRY_STOP_AFTER_ATTEMPT: Final[int] = 3
ARXIV_PROC_RETRY_WAIT_EXPONENTIAL_MULTIPLIER: Final[int] = 1
ARXIV_PROC_RETRY_WAIT_EXPONENTIAL_MIN_SECONDS: Final[int] = 2
ARXIV_PROC_RETRY_WAIT_EXPONENTIAL_MAX_SECONDS: Final[int] = 10

# Logging and Output paths for the arXiv script (these are subdirectories/filenames within PROJECT_ROOT)
ARXIV_PROC_LOG_DIR_NAME: Final[str] = "logs_arxiv_collection" 
ARXIV_PROC_LOG_FILENAME: Final[str] = "arxiv_data_collection_run.log"
ARXIV_PROC_OUTPUT_DATA_SUBDIR: Final[str] = "arxiv_dataset" 
ARXIV_PROC_OUTPUT_FILENAME: Final[str] = "arxiv_papers_collected_feed.jsonl"

# Default log level for the arXiv script (can be overridden by CLI)
ARXIV_PROC_LOG_LEVEL: Final[str] = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Optional: Maximum runtime for the script in seconds (None for unlimited).
ARXIV_PROC_MAX_RUNTIME_SECONDS: Final[Optional[int]] = None 

# Interval for logging progress updates during processing.
ARXIV_PROC_PROGRESS_LOG_INTERVAL: Final[int] = 300

# ==============================================================================
# --- Utility & Checks ---
# ==============================================================================
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Data directory checked/created: {DATA_DIR}")
except OSError as e:
    logger.error(f"Could not create data directory {DATA_DIR}: {e}", exc_info=True)
except Exception as e:
    logger.error(f"Unexpected error checking/creating data directory {DATA_DIR}: {e}", exc_info=True)

# --- API Key / Configuration Checks ---
def run_config_checks():
    """Runs checks for essential configuration and logs warnings."""
    config_ok = True
    providers_checked = set()
    at_least_one_provider_ok = False
    logger.info("Running configuration checks...")

    if not LLM_PROVIDER_ORDER:
        logger.error("LLM_PROVIDER_ORDER list is empty. No LLM providers specified.")
        config_ok = False

    for provider in LLM_PROVIDER_ORDER:
        if provider in providers_checked: continue
        providers_checked.add(provider)
        provider_models = []
        api_key = None
        key_name = ""
        provider_ok = False

        if provider == "google":
            provider_models = LLM_GOOGLE_MODELS
            api_key = GOOGLE_API_KEY
            key_name = "GOOGLE_API_KEY"
        elif provider == "groq":
            provider_models = LLM_GROQ_MODELS
            api_key = GROQ_API_KEY
            key_name = "GROQ_API_KEY"
        elif provider == "deepseek":
            provider_models = LLM_DEEPSEEK_MODELS
            api_key = DEEPSEEK_API_KEY
            key_name = "DEEPSEEK_API_KEY"
        else:
            logger.error(f"Unsupported provider found in LLM_PROVIDER_ORDER: '{provider}'.")
            config_ok = False
            continue

        if not provider_models:
             logger.warning(f"Provider '{provider}' is in order, but its model list (e.g., LLM_{provider.upper()}_MODELS) is empty.")
        elif not api_key:
             logger.warning(f"Provider '{provider}' is configured with models, but its API key ({key_name}) is missing in .env.")
             if len(LLM_PROVIDER_ORDER) == 1 or provider == LLM_PROVIDER_ORDER[0]:
                  config_ok = False
        else:
             logger.info(f"Provider '{provider}' seems configured (API key found, models listed: {provider_models}).")
             provider_ok = True
             at_least_one_provider_ok = True

    # General Checks
    if EMBEDDING_DIM <= 0:
         logger.error(f"Invalid EMBEDDING_DIM ({EMBEDDING_DIM}). Must be positive.")
         config_ok = False

    if not at_least_one_provider_ok:
        logger.critical("No LLM providers appear to be configured correctly with both models and API keys.")
        config_ok = False

    if config_ok:
         logger.info(f"Configuration checks passed (at least one provider seems usable).")
    else:
         logger.critical("One or more critical configuration checks failed. Functionality will be impaired. Please review errors/warnings and check your .env file and provider model lists.")
         # Print a prominent message if critical config is missing
         print("\n" + "="*60 + "\n"
               "🚨 CRITICAL CONFIGURATION ERROR 🚨\n"
               "   No LLM provider seems fully configured with both models and an API key,\n"
               "   or an essential setting (like EMBEDDING_DIM) is invalid.\n"
               f"   Please check logs above and ensure the required keys (e.g., GOOGLE_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY)\n"
               f"   are correctly set in '{DOTENV_PATH}' and that model lists (e.g., LLM_GOOGLE_MODELS, LLM_DEEPSEEK_MODELS)\n"
               "   are populated for the providers listed in LLM_PROVIDER_ORDER.\n"
               "   The application may not function correctly.\n"
               + "="*60 + "\n", file=sys.stderr)

# Run checks when the module is imported
run_config_checks()

logger.info("Configuration loading complete.")
