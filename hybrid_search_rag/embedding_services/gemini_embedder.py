# hybrid_search_rag/embedding_services/gemini_embedder.py
"""Handles encoding text using the Google Generative AI (Gemini) API."""

import logging
import time
import typing
import numpy as np
import math
import concurrent.futures
import google.generativeai as genai
from tenacity import stop_after_attempt, wait_exponential, retry_if_exception_type, retry

# Assuming config is accessible via relative import
from .. import config

logger = logging.getLogger(__name__)

# --- Configuration ---
GOOGLE_API_KEY = config.GOOGLE_API_KEY

# Model and batching settings
GEMINI_EMBEDDING_MODEL_ID = getattr(config, 'GEMINI_EMBEDDING_MODEL', "models/gemini-embedding-001")
GEMINI_API_BATCH_LIMIT = getattr(config, 'GEMINI_API_BATCH_LIMIT', 100)
API_DELAY_SECONDS = getattr(config, 'EMBEDDING_API_DELAY', 0.1)

try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
    RETRYABLE_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable)
except ImportError:
    logger.warning("google-api-core not installed. Using broad Exception for retries.")
    RETRYABLE_EXCEPTIONS = (Exception,)

def retry_with_exponential_backoff(
    stop_config=stop_after_attempt(3),
    wait_config=wait_exponential(multiplier=1, min=2, max=10),
    retry_condition_config=retry_if_exception_type(RETRYABLE_EXCEPTIONS)
):
    """Decorator to retry a function with exponential backoff."""
    def decorator(func):
        return retry(stop=stop_config, wait=wait_config, retry=retry_condition_config)(func)
    return decorator

class EmbeddingModel:
    """Wraps the Google Generative AI (Gemini) API for encoding text."""

    _is_configured = False

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL_ID):
        """Initializes the wrapper and configures the genai library if needed."""
        self.model_name = model_name
        logger.info(f"Initializing EmbeddingModel for Gemini API (model: {self.model_name})")

        api_key = config.GOOGLE_API_KEY
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found. Cannot initialize Gemini EmbeddingModel.")

        if not EmbeddingModel._is_configured:
            try:
                logger.info("Configuring Google Generative AI library...")
                genai.configure(api_key=api_key) # type: ignore
                EmbeddingModel._is_configured = True
                logger.info("Google Generative AI library configured.")
            except Exception as e:
                logger.error(f"Failed to configure Google Generative AI library: {e}", exc_info=True)
                EmbeddingModel._is_configured = False
                raise RuntimeError(f"Google Generative AI Configuration Failed: {e}") from e

    @retry_with_exponential_backoff()
    def _embed_batch(self, batch_texts: list[str], task_type: str) -> typing.Optional[list[list[float]]]:
        """Handles the actual API call for embedding a batch of texts."""
        try:
            logger.debug(f"Sending batch ({len(batch_texts)} items) to Gemini API...")
            result = genai.embed_content( # type: ignore
                model=self.model_name,
                content=batch_texts,
                task_type=task_type
            )
            if isinstance(result, dict) and 'embedding' in result and isinstance(result['embedding'], list):
                if len(result['embedding']) == len(batch_texts):
                    return typing.cast(list[list[float]], result['embedding'])
                else:
                    logger.error(f"Mismatched embedding count: API returned {len(result['embedding'])} for a batch of {len(batch_texts)}.")
                    return None
            else:
                logger.error(f"Unexpected API response format: {type(result)}.")
                return None
        except Exception as e:
            logger.error(f"Error during Gemini embed_content API call: {e}", exc_info=True)
            raise

    def encode(self,
             texts: typing.Union[str, list[str]],
             batch_size: typing.Optional[int] = None,
             task_type: str = "RETRIEVAL_DOCUMENT",
             max_workers: int = 10
             ) -> typing.Optional[np.ndarray]:
        """
        Encodes text(s) into vector embeddings using parallel batch processing.
        """
        if not EmbeddingModel._is_configured:
            logger.error("Gemini API not configured. Cannot encode.")
            return None

        texts_list = [texts] if isinstance(texts, str) else [str(t) if t is not None else "" for t in texts]
        original_len = len(texts_list)
        texts_list = [t for t in texts_list if t.strip()]
        if (removed_count := original_len - len(texts_list)) > 0:
            logger.warning(f"Removed {removed_count} empty/None strings.")
        if not texts_list:
            return np.array([], dtype=np.float32)

        effective_batch_size = batch_size if batch_size and batch_size > 0 else GEMINI_API_BATCH_LIMIT
        num_texts_to_encode = len(texts_list)
        num_batches = math.ceil(num_texts_to_encode / effective_batch_size)
        
        logger.info(f"Encoding {num_texts_to_encode} texts in {num_batches} batches (size={effective_batch_size}, max_workers={max_workers}).")

        all_embedding_values: list[typing.Optional[list[list[float]]]] = [None] * num_batches
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch_index = {}
            for i in range(num_batches):
                batch_start = i * effective_batch_size
                batch_end = min((i + 1) * effective_batch_size, num_texts_to_encode)
                current_batch_texts = texts_list[batch_start:batch_end]
                
                if not current_batch_texts:
                    continue

                logger.debug(f"Submitting batch {i + 1}/{num_batches} for processing.")
                future = executor.submit(self._embed_batch, current_batch_texts, task_type)
                future_to_batch_index[future] = i
                time.sleep(API_DELAY_SECONDS)

            for future in concurrent.futures.as_completed(future_to_batch_index):
                batch_index = future_to_batch_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_embedding_values[batch_index] = result
                        logger.debug(f"Successfully completed and stored batch {batch_index + 1}/{num_batches}.")
                    else:
                        logger.error(f"Batch {batch_index + 1}/{num_batches} failed and returned None.")
                except Exception as exc:
                    logger.error(f"Batch {batch_index + 1} generated an exception: {exc}", exc_info=True)

        final_embeddings = [emb for batch in all_embedding_values if batch is not None for emb in batch]

        if len(final_embeddings) != num_texts_to_encode:
            logger.error(f"Encoding mismatch: Expected {num_texts_to_encode} embeddings, but received {len(final_embeddings)}.")
            return None
        
        logger.info(f"Successfully encoded all {len(final_embeddings)} texts.")
        try:
            embeddings_np = np.array(final_embeddings, dtype=np.float32)
            logger.info(f"Final embedding shape: {embeddings_np.shape}")
            return embeddings_np
        except ValueError as e:
            logger.error(f"Failed to convert to NumPy array, likely due to inconsistent dimensions: {e}", exc_info=True)
            return None