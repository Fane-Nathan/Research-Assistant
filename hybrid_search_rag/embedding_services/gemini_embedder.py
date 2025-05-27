# hybrid_search_rag/embedding_services/gemini_embedder.py
"""Handles encoding text using the Google Generative AI (Gemini) API."""

import logging
import time
from typing import List, Union, Optional
import numpy as np
import math
import google.generativeai as genai
from tenacity import stop_after_attempt, wait_exponential, retry_if_exception_type, stop, wait, retry

# Assuming config is accessible via relative import
from .. import config

logger = logging.getLogger(__name__)

# --- Configuration ---
try:
    GOOGLE_API_KEY = config.GOOGLE_API_KEY
    if not GOOGLE_API_KEY:        
        raise RuntimeError("GOOGLE_API_KEY not set in config or .env file.")    
except (AttributeError, ValueError) as e: # AttributeError can be raised by getattr if config itself is missing
    raise RuntimeError(f"Configuration Error: {e}. Cannot initialize Gemini EmbeddingModel.") from e

# Use default model ID or override from config
GEMINI_EMBEDDING_MODEL_DEFAULT = "models/text-embedding-004" # Renamed for clarity
GEMINI_EMBEDDING_MODEL_ID = getattr(config, 'GEMINI_EMBEDDING_MODEL', GEMINI_EMBEDDING_MODEL_DEFAULT)

# Batching and Delay settings from config or defaults
GEMINI_API_BATCH_LIMIT = getattr(config, 'GEMINI_API_BATCH_LIMIT', 100) # Default batch limit
API_DELAY_SECONDS = getattr(config, 'EMBEDDING_API_DELAY', 0.1) # Default delay

# Define exceptions that might warrant retries (e.g., rate limits, temporary server errors)
try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable # type: ignore
    RETRYABLE_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable)
except ImportError:
    logger.warning("google-api-core not installed. Using broad Exception for retries, which might not be ideal.")
    RETRYABLE_EXCEPTIONS = (Exception,) # Fallback to generic Exception for retry logic


def retry_with_exponential_backoff(
    stop_config: stop.stop_base = stop_after_attempt(3), # Renamed for clarity
    wait_config: wait.wait_base = wait_exponential(multiplier=1, min=2, max=10), # Renamed
    retry_condition_config = retry_if_exception_type(RETRYABLE_EXCEPTIONS) # Renamed
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        stop_config: Tenacity stop condition (e.g., after a maximum number of attempts).
        wait_config: Tenacity wait strategy (e.g., exponential backoff).
        retry_condition_config: Tenacity retry condition (e.g., based on exception type).

    Returns:
        Decorated function.
    """
    def decorator(func):
        return retry(stop=stop_config, wait=wait_config, retry=retry_condition_config)(func)
    return decorator

class EmbeddingModel:
    """Wraps the Google Generative AI (Gemini) API for encoding text."""

    _is_configured = False # Class flag to track if genai.configure has been called

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL_ID):
        """Initializes the wrapper and configures the genai library if needed."""
        self.model_name = model_name
        logger.info(f"Initializing EmbeddingModel for Gemini API (model: {self.model_name})")

        if not GOOGLE_API_KEY: # Should have been caught by module-level check, but good for robustness
            raise RuntimeError("GOOGLE_API_KEY not found. Cannot initialize Gemini EmbeddingModel.")

        # Configure the genai library only once per class load
        if not EmbeddingModel._is_configured:
            try:
                logger.info("Configuring Google Generative AI library...")
                # Pylance might warn about `genai.configure` not being exported.
                # This is likely a type stub issue; genai.configure is standard.
                genai.configure(api_key=GOOGLE_API_KEY) # type: ignore
                EmbeddingModel._is_configured = True
                logger.info("Google Generative AI library configured.")
            except Exception as e:
                logger.error(f"Failed to configure Google Generative AI library: {e}", exc_info=True)
                EmbeddingModel._is_configured = False # Ensure flag remains false on error
                # Re-raise as a RuntimeError to indicate a critical setup failure
                raise RuntimeError(f"Google Generative AI Configuration Failed: {e}") from e

    @retry_with_exponential_backoff()
    def _embed_batch(self, batch_texts: List[str], task_type: str) -> Optional[List[List[float]]]:
        """
        Handles the actual API call for embedding a batch of texts,
        including error handling and retries.
        Pylance might flag the return type. However, for a list input to `content`,
        `result['embedding']` is expected to be `List[List[float]]`.
        """
        try:
            logger.debug(f"Sending batch ({len(batch_texts)} items) to Gemini API for model {self.model_name}...")
            # Pylance might warn about `genai.embed_content` not being exported.
            # This is likely a type stub issue; genai.embed_content is standard.
            result = genai.embed_content( # type: ignore
                model=self.model_name,
                content=batch_texts, # Critical: content is List[str]
                task_type=task_type
            )

            # Validate and extract embeddings
            if isinstance(result, dict) and 'embedding' in result and isinstance(result['embedding'], list):
                # Ensure all elements in the 'embedding' list are also lists (of floats)
                if not result['embedding'] or all(isinstance(emb, list) for emb in result['embedding']):
                    if len(result['embedding']) == len(batch_texts):
                        # Type cast for clarity, though API should guarantee List[List[float]] here
                        return typing.cast(List[List[float]], result['embedding'])
                    else:
                        logger.error(f"Gemini API returned {len(result['embedding'])} embeddings for a batch of {len(batch_texts)}.")
                        return None
                else:
                    logger.error(f"Gemini API returned embeddings with unexpected inner structure. Expected List[List[float]], got List[{type(result['embedding'][0]) if result['embedding'] else 'Unknown'}].")
                    return None
            else:
                logger.error(f"Gemini API returned unexpected response format: {type(result)}. Expected dict with 'embedding' key.")
                return None
        except Exception as e:
            logger.error(f"Error during Gemini embed_content API call for model {self.model_name}: {e}", exc_info=True)
            raise # Re-raise exception to allow tenacity retry mechanism

    def encode(self,
               texts: Union[str, List[str]],               
               batch_size: Optional[int] = None,              
               task_type: str = "RETRIEVAL_DOCUMENT" # Default task type
               ) -> Optional[np.ndarray]:        
        """
        Encodes text(s) into vector embeddings using the Gemini API with batching and retries.

        Args:
            texts: A single string or a list of strings to embed.
            batch_size: Max number of texts per API call. Uses GEMINI_API_BATCH_LIMIT from config if None.
            task_type: The task type for the embedding (e.g., 'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY').

        Returns:
            A NumPy array of embeddings (np.float32), or None if encoding failed.
        """
        if not EmbeddingModel._is_configured:            
            logger.error("Gemini API not configured. Cannot encode.")
            return None

        # --- Input Validation and Preparation ---
        if isinstance(texts, str):
             texts_list = [texts]
        elif isinstance(texts, list):
             texts_list = [str(t) if t is not None else "" for t in texts] # Convert all to str, handle None
        else: # Attempt conversion for other types
             texts_list = [str(texts)]

        original_len = len(texts_list)
        texts_list = [t for t in texts_list if t.strip()] # Remove empty strings
        removed_count = original_len - len(texts_list)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} empty or None string(s) from input. They will be skipped.")

        if not texts_list: # If all texts were empty or None
            logger.warning("Input contains no valid text to encode after validation.")
            return np.array([], dtype=np.float32)

        # --- Batch Processing ---
        all_embedding_values: List[List[float]] = []
        num_texts_to_encode = len(texts_list)
        
        effective_batch_size = batch_size if batch_size is not None and batch_size > 0 else GEMINI_API_BATCH_LIMIT
        num_batches = math.ceil(num_texts_to_encode / effective_batch_size)

        logger.info(f"Encoding {num_texts_to_encode} texts using Gemini '{self.model_name}' in {num_batches} batches (size={effective_batch_size}, task={task_type}).")

        for i in range(num_batches):
            batch_start_index = i * effective_batch_size
            batch_end_index = min((i + 1) * effective_batch_size, num_texts_to_encode)
            current_batch_texts = texts_list[batch_start_index:batch_end_index]
            
            logger.debug(f"Processing batch {i + 1}/{num_batches} (indices {batch_start_index}-{batch_end_index-1}). Size: {len(current_batch_texts)}")

            if not current_batch_texts: # Should not happen if logic is correct
                logger.warning(f"Skipping empty batch {i+1}/{num_batches}.")
                continue

            batch_embeddings = self._embed_batch(current_batch_texts, task_type)

            if batch_embeddings is None: # Indicates failure after retries
                logger.error(f"Failed to embed batch {i+1}/{num_batches}. Aborting encoding process.")
                return None 
            
            all_embedding_values.extend(batch_embeddings)
            logger.debug(f"Successfully embedded batch {i+1}/{num_batches}. Received {len(batch_embeddings)} embeddings.")

            if i < num_batches - 1 and API_DELAY_SECONDS > 0:
                logger.debug(f"Waiting {API_DELAY_SECONDS}s before next batch...")
                time.sleep(API_DELAY_SECONDS)
                
        # --- Final Conversion and Validation ---
        if len(all_embedding_values) == num_texts_to_encode:
            logger.info(f"Successfully encoded all {num_texts_to_encode} texts.")
            try:
                # Ensure all sub-lists (embeddings) have the same dimension before creating NumPy array
                if all_embedding_values:
                    first_embedding_dim = len(all_embedding_values[0])
                    if not all(len(emb) == first_embedding_dim for emb in all_embedding_values):
                        logger.error("Inconsistent embedding dimensions returned by API across batches or items.")
                        # Log lengths of a few problematic embeddings for debugging
                        for idx, emb_val in enumerate(all_embedding_values):
                            if len(emb_val) != first_embedding_dim:
                                logger.error(f"Embedding at index {idx} has length {len(emb_val)}, expected {first_embedding_dim}.")
                                if idx > 5: break # Log a few examples
                        return None
                
                embeddings_np = np.array(all_embedding_values, dtype=np.float32)
                
                # Final shape check
                if embeddings_np.ndim == 2 and embeddings_np.shape[0] == num_texts_to_encode:
                     logger.info(f"Final embedding shape: {embeddings_np.shape}")
                     return embeddings_np
                elif num_texts_to_encode == 0 and embeddings_np.shape == (0,): # Handle case of empty input resulting in empty array
                     logger.info("Encoded 0 texts, returning empty_array.shape=(0,)")
                     return embeddings_np # This is np.array([]) which has shape (0,)
                else: # Should be caught by inconsistent dim check, but as a safeguard
                     logger.error(f"Final NumPy array shape is incorrect: {embeddings_np.shape}. Expected ({num_texts_to_encode}, embedding_dim).")
                     return None
            except ValueError as e: # Error during np.array conversion
                 logger.error(f"Failed to convert embeddings to NumPy array: {e}. This might indicate inconsistent embedding dimensions.", exc_info=True)
                 return None
        else: # Should ideally not be reached if batch processing logic is correct
            logger.error(f"Encoding mismatch: Expected {num_texts_to_encode} embeddings, but received {len(all_embedding_values)} after batch processing.")
            return None

# Example of adding typing import for cast
import typing
