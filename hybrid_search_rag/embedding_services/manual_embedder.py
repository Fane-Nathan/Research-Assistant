# hybrid_search_rag/retrieval/embedding_model.py
"""Handles loading the Sentence-BERT model and encoding text."""

from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union, Optional
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wraps the SentenceTransformer model for encoding text."""
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls, model_name: str):
        """Implement Singleton pattern to load model only once."""
        if cls._instance is None:
            logger.info("Creating new EmbeddingModel instance.")
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._model_name = model_name
            cls._model = None
            cls._instance._check_gpu()
        elif cls._model_name != model_name:
            logger.warning(f"EmbeddingModel already initialized with {cls._model_name}. Ignoring new name {model_name}.")
        return cls._instance

    def __init__(self, model_name: str):
        """Initializes the EmbeddingModel attributes if not already set."""
        if not hasattr(self, 'model_name') or self.model_name is None:
             self.model_name = model_name
        if not hasattr(self, 'model') or self.model is None:
            self.model = self._model 


    def _check_gpu(self):
        """Checks for TensorFlow GPU availability and sets memory growth."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"TensorFlow found GPUs: {gpus}")
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Enabled memory growth for GPUs.")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth (likely already initialized): {e}")
            else:
                logger.info("No GPU found by TensorFlow, using CPU for potential TF ops.")
        except Exception as e:
            logger.error(f"Error during GPU check: {e}")

    def load_model(self):
        """Loads the SentenceTransformer model if not already loaded."""
        if EmbeddingModel._model is None: 
            if not EmbeddingModel._model_name:
                 logger.error("Cannot load model: model_name not set.")
                 return
            try:
                logger.info(f"Loading SentenceTransformer model: {EmbeddingModel._model_name}...")
                EmbeddingModel._model = SentenceTransformer(EmbeddingModel._model_name, device=None)
                self.model = EmbeddingModel._model 
                detected_device = EmbeddingModel._model.device
                EmbeddingModel._device = detected_device
                self.device = detected_device
                logger.info(f"Model {EmbeddingModel._model_name} loaded successfully on device: {EmbeddingModel._model.device}")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{EmbeddingModel._model_name}': {e}", exc_info=True)
                EmbeddingModel._model = None
                self.model = None
                if hasattr(self, 'device'): self.device = None
                raise

    def get_model(self) -> Optional[SentenceTransformer]:
        """Returns the loaded model instance, loading it if necessary."""
        if self.model is None:
             self.load_model()
        return self.model


    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> Optional[np.ndarray]:
        """Encodes text(s) into vector embeddings using the loaded model."""
        model_instance = self.get_model()
        if model_instance is None:
            logger.error("Embedding model is not available for encoding.")
            return None

        if isinstance(texts, list):
            valid_texts = [str(t) if t is not None else "" for t in texts]
            if not any(t.strip() for t in valid_texts):
                logger.warning("Input list contains only empty or None strings after validation. Encoding empty strings.")
        elif isinstance(texts, str):
            valid_texts = texts
        elif texts is None:
             valid_texts = ""
        else:
            valid_texts = str(texts)

        if not valid_texts and isinstance(valid_texts, str):
             logger.warning("Input text is empty or None. Encoding empty string.")

        try:
            num_items = 1 if isinstance(valid_texts, str) else len(valid_texts)
            logger.info(f"Encoding {num_items} item(s) using {self.model_name}...")
            embeddings = model_instance.encode(
                texts,
                batch_size=batch_size, 
                show_progress_bar=True, 
                convert_to_tensor=False, 
            )
            logger.info("Encoding complete.")
            return np.array(embeddings, dtype=np.float32) if not isinstance(embeddings, np.ndarray) else embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"An error occurred during text encoding: {e}", exc_info=True)
            return None