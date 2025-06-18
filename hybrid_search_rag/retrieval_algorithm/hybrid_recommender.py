# hybrid_search_rag/retrieval_algorithm/hybrid_recommender.py
"""Handles hybrid recommendation using semantic search (embeddings) and keyword search (BM25),
fusing the results using Reciprocal Rank Fusion (RRF)."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import logging
import nltk
import ssl
from nltk.corpus import stopwords
import string
import sys

from ..embedding_services.gemini_embedder import EmbeddingModel as GeminiEmbedder
from ..query_processor import QueryProcessor


@dataclass
class RecommendationParams:
    """Parameters for recommendation."""
    semantic_candidates: int
    keyword_candidates: int
    fusion_k: int
    top_n_final: int
    expand_synonyms: bool = True
    top_n_rerank: int = 5

logger = logging.getLogger(__name__)

class NltkManager:
    """Manages NLTK data and tokenization, ensuring data is available."""
    NLTK_STOPWORDS: Optional[set[str]] = None
    NLTK_DATA_AVAILABLE: Dict[str, bool] = {'punkt': False, 'stopwords': False}
    _nltk_checked_init = False

    def __init__(self):
        """Initialize and ensure NLTK data is checked/loaded."""
        if not NltkManager._nltk_checked_init:
            NltkManager._check_and_load_nltk_data()
            NltkManager._nltk_checked_init = True

    @classmethod
    def _check_and_load_nltk_data(cls):
        """
        Checks for required NLTK data ('punkt' for tokenization, 'stopwords').
        Attempts to download them if missing.
        """
        data_to_check = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
        needs_download = []
        initial_availability = cls.NLTK_DATA_AVAILABLE.copy()

        logger.info("Performing NLTK data check...")
        for name, path_fragment in data_to_check.items():
            try:
                nltk.data.find(path_fragment)
                cls.NLTK_DATA_AVAILABLE[name] = True
            except LookupError:
                cls.NLTK_DATA_AVAILABLE[name] = False
                if name not in needs_download:
                    needs_download.append(name)
                    logger.warning(f"NLTK data '{name}' not found. Will attempt download.")
            except Exception as e:
                logger.error(f"Error checking NLTK data '{name}': {e}")
                cls.NLTK_DATA_AVAILABLE[name] = False


        if needs_download:
            logger.info(f"Attempting to download missing NLTK data: {', '.join(needs_download)}...")
            download_success_flags: Dict[str, bool] = {}
            try:
                # Attempt to bypass SSL verification issues if they occur during download
                # This is a common workaround for NLTK downloads in some environments.
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass # SSL context workaround not needed or not applicable
                else:
                    ssl._create_default_https_context = _create_unverified_https_context

                for name in needs_download:
                    print(f"Downloading NLTK package: {name}...", file=sys.stderr) 
                    if nltk.download(name, quiet=True): 
                        logger.info(f"Successfully initiated download for NLTK data '{name}'. Verifying...")
                        try: 
                            nltk.data.find(data_to_check[name])
                            cls.NLTK_DATA_AVAILABLE[name] = True
                            download_success_flags[name] = True
                            logger.info(f"NLTK data '{name}' verified successfully after download.")
                        except LookupError:
                             logger.error(f"Verification failed for NLTK data '{name}' even after download attempt. Package might be corrupted or not found where expected.")
                             cls.NLTK_DATA_AVAILABLE[name] = False
                             download_success_flags[name] = False
                    else: 
                        logger.error(f"NLTK download command failed for '{name}'.")
                        cls.NLTK_DATA_AVAILABLE[name] = False
                        download_success_flags[name] = False
            except Exception as e:
                logger.error(f"NLTK download process failed: {e}", exc_info=True)
                for name_in_error_case in needs_download: 
                     if name_in_error_case not in download_success_flags:
                          cls.NLTK_DATA_AVAILABLE[name_in_error_case] = False
            
            final_failed_downloads = [name for name, success in download_success_flags.items() if not success]
            for name_to_check in needs_download:
                if name_to_check not in download_success_flags and name_to_check not in final_failed_downloads:
                    final_failed_downloads.append(name_to_check)

            if final_failed_downloads:
                 print(f"\nERROR: Failed to download/verify required NLTK data: {', '.join(final_failed_downloads)}.", file=sys.stderr)
                 print("Please try installing them manually in your Python environment:", file=sys.stderr)
                 for name_failed in final_failed_downloads:
                     print(f"  >>> import nltk; nltk.download('{name_failed}')", file=sys.stderr)
                 print("Keyword search and tokenization functionality may be impaired.", file=sys.stderr)

        if cls.NLTK_DATA_AVAILABLE['stopwords'] and cls.NLTK_STOPWORDS is None:
            try:
                cls.NLTK_STOPWORDS = set(stopwords.words('english'))
                logger.info(f"Loaded {len(cls.NLTK_STOPWORDS)} NLTK English stopwords.")
            except Exception as e:
                logger.error(f"Failed to load NLTK stopwords even though data seems available: {e}", exc_info=True)
                cls.NLTK_STOPWORDS = set() 
        elif not cls.NLTK_DATA_AVAILABLE['stopwords'] and initial_availability['stopwords']:
             logger.warning("NLTK stopwords data became unavailable after initial check. Using empty set for stopwords.")
             cls.NLTK_STOPWORDS = set()
        elif cls.NLTK_STOPWORDS is None: 
            logger.warning("NLTK stopwords data not available. Stopword removal will be skipped if requested.")
            cls.NLTK_STOPWORDS = set()


    @classmethod
    def tokenize_text(cls, text: str, remove_stopwords: bool = True, min_word_length: int = 2) -> List[str]:
        """
        Tokenizes text, optionally removes stopwords and short words.
        Ensures NLTK 'punkt' (tokenizer) data is available.
        """
        if not isinstance(text, str):
            logger.warning(f"Attempted to tokenize non-string input: {type(text)}. Returning empty list.")
            return []

        if not cls.NLTK_DATA_AVAILABLE['punkt']:
            logger.warning("NLTK 'punkt' (tokenizer) data not available. Attempting to re-check/load...")
            cls._check_and_load_nltk_data() 
            if not cls.NLTK_DATA_AVAILABLE['punkt']:
                logger.error("NLTK 'punkt' data is unavailable even after re-check! Cannot tokenize text. Returning empty list.")
                return []

        if remove_stopwords and cls.NLTK_STOPWORDS is None:
                 logger.debug("Stopwords requested but NLTK_STOPWORDS is None; re-checking data.")
                 cls._check_and_load_nltk_data()

        try:
            processed_text = text.lower()
            processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
            tokens = nltk.word_tokenize(processed_text)

            if remove_stopwords and cls.NLTK_STOPWORDS: 
                tokens = [word for word in tokens if word not in cls.NLTK_STOPWORDS]
            elif remove_stopwords:
                logger.debug("Stopword removal requested, but no stopwords loaded. Skipping removal.")

            return [word for word in tokens if len(word) >= min_word_length]
        except Exception as e: 
            logger.error(f"Tokenization failed for text snippet: '{text[:50]}...': {e}", exc_info=True)
            return []

class HybridRecommender:
    """Orchestrates hybrid search using embeddings and BM25, fused with RRF."""

    def __init__(self, embed_model: GeminiEmbedder, enable_query_preprocessing: bool = True):
        """
        Initializes with a pre-configured EmbeddingModel instance.
        
        Args:
            embed_model: Pre-configured EmbeddingModel instance
            enable_query_preprocessing: Whether to enable query preprocessing and expansion
        """
        self.embed_model = embed_model
        self.nltk_manager = NltkManager() 
        self.enable_query_preprocessing = enable_query_preprocessing
        
        # Initialize QueryProcessor if preprocessing is enabled
        if self.enable_query_preprocessing:
            self.query_processor = QueryProcessor(use_nltk_expansion=True)
            logger.info("HybridRecommender initialized with query preprocessing enabled.")
        else:
            self.query_processor = None
            logger.info("HybridRecommender initialized with query preprocessing disabled.")
            
        logger.info(f"HybridRecommender initialized with embedding model: {type(embed_model).__name__}")

    def _validate_embeddings(self, query_embedding: Optional[np.ndarray], resource_embeddings: Optional[np.ndarray]) -> bool:
        """Helper function to validate embeddings for semantic search."""
        if query_embedding is None:
            logger.warning("Embeddings validation failed: query_embedding is None.")
            return False
        if resource_embeddings is None:
            logger.warning("Embeddings validation failed: resource_embeddings is None.")
            return False
            
        if query_embedding.size == 0:
            logger.warning(f"Embeddings validation failed: query_embedding size is 0.")
            return False
        if resource_embeddings.size == 0:
            logger.warning(f"Embeddings validation failed: resource_embeddings size is 0.")
            return False

        query_embedding_2d = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
        
        if query_embedding_2d.ndim != 2 or query_embedding_2d.shape[0] != 1:
            logger.error(f"Invalid query embedding shape after reshape: {query_embedding_2d.shape}. Expected (1, D). Original: {query_embedding.shape}.")
            return False
        if resource_embeddings.ndim != 2:
            logger.error(f"Invalid resource embeddings shape: {resource_embeddings.shape}. Expected (N, D).")
            return False
        if resource_embeddings.shape[1] != query_embedding_2d.shape[1]:
            logger.error(f"Embeddings dimension mismatch: Resource embeddings dim {resource_embeddings.shape[1]}, Query embedding dim {query_embedding_2d.shape[1]}.")
            return False
        return True

    def _validate_keyword_search_inputs(self, query: str, bm25_index: Optional[BM25Okapi]) -> Tuple[bool, List[str]]:
        """Helper function to validate inputs for keyword search."""
        if bm25_index is None:
            logger.info("BM25 index is None. Skipping keyword search.")
            return False, []

        tokenized_query = self.nltk_manager.tokenize_text(query, remove_stopwords=True, min_word_length=2)
        if not tokenized_query:
            logger.info(f"Query '{query[:50]}...' became empty after tokenization. Skipping keyword search.")
            return False, []
        return True, tokenized_query

    def _semantic_search(self, query_embedding: Optional[np.ndarray], resource_embeddings: Optional[np.ndarray], top_n: int) -> List[Tuple[int, float]]:
        """Performs semantic search using cosine similarity."""
        if not self._validate_embeddings(query_embedding, resource_embeddings):
            return []
        
        assert query_embedding is not None, "Query embedding should be validated by _validate_embeddings"
        assert resource_embeddings is not None, "Resource embeddings should be validated by _validate_embeddings"

        if top_n <= 0:
            logger.warning(f"Semantic search top_n is {top_n}, must be positive. Returning empty list.")
            return []
        try:
            query_embedding_2d = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
            
            similarities = cosine_similarity(query_embedding_2d, resource_embeddings)[0]
            
            num_docs_in_corpus = resource_embeddings.shape[0]
            num_results_to_fetch = min(top_n, num_docs_in_corpus)

            if num_results_to_fetch <= 0:
                return []
                
            top_indices_unsorted = np.argpartition(similarities, -num_results_to_fetch)[-num_results_to_fetch:]
            top_indices_sorted = top_indices_unsorted[np.argsort(similarities[top_indices_unsorted])[::-1]]

            results = [(int(i), float(similarities[i])) for i in top_indices_sorted]
            logger.debug(f"Semantic search returned {len(results)} candidates for top_n={top_n}.")
            return results
        except Exception as e:
             logger.error(f"Error during semantic search: {e}", exc_info=True)
             return []

    def _keyword_search(self, query: str, bm25_index: Optional[BM25Okapi], num_docs_in_corpus: int, top_n: int) -> List[Tuple[int, float]]:
        """Performs keyword search using the BM25 index."""
        valid_inputs, tokenized_query = self._validate_keyword_search_inputs(query, bm25_index)
        if not valid_inputs: 
            return []
        
        assert bm25_index is not None, "BM25 index should be validated by _validate_keyword_search_inputs"

        if top_n <= 0:
            logger.warning(f"Keyword search top_n is {top_n}, must be positive. Returning empty list.")
            return []
        try:
            doc_scores = bm25_index.get_scores(tokenized_query)

            num_results_to_fetch = min(top_n, len(doc_scores))
            if num_results_to_fetch <= 0:
                return []

            top_indices_unsorted = np.argpartition(doc_scores, -num_results_to_fetch)[-num_results_to_fetch:]
            top_indices_sorted = top_indices_unsorted[np.argsort(doc_scores[top_indices_unsorted])[::-1]]
            
            results = [(int(i), float(doc_scores[i])) for i in top_indices_sorted]
            logger.debug(f"Keyword search returned {len(results)} candidates for top_n={top_n}.")
            return results
        except Exception as e:
             logger.error(f"Error during keyword search: {e}", exc_info=True)
             return []
    
    # Reciprocal Rank Fusion (RRF) implementation

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Tuple[int, float]]], k_rrf: int = 60) -> Dict[int, float]:
        """Combines multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
        fused_scores: Dict[int, float] = {}
        if not ranked_lists: 
            return fused_scores

        logger.debug(f"Performing RRF (k_rrf={k_rrf}) on {len(ranked_lists)} lists...")
        for rank_list in ranked_lists:
            if not rank_list: continue
            seen_indices_in_this_list = set()
            for rank, item in enumerate(rank_list):
                try:
                    doc_index, _score = item
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping malformed item in rank_list during RRF: {item}. Error: {e}")
                    continue

                if isinstance(doc_index, (int, np.integer)) and doc_index >= 0:
                    doc_index_int = int(doc_index)
                    if doc_index_int not in seen_indices_in_this_list:
                        # RRF formula: 1 / (k + rank). Rank is 0-indexed here, so rank + 1 for 1-based ranking.
                        fused_scores[doc_index_int] = fused_scores.get(doc_index_int, 0.0) + (1.0 / (k_rrf + rank + 1))
                        seen_indices_in_this_list.add(doc_index_int)
                else:
                    logger.warning(f"Skipping invalid doc_index type ({type(doc_index)}) or value ({doc_index}) during RRF.")
        logger.debug(f"RRF fusion generated {len(fused_scores)} unique document scores.")
        return fused_scores

    def recommend(self, query: str, resource_metadata: List[Dict[str, Any]],
                  resource_embeddings: Optional[np.ndarray], bm25_index: Optional[BM25Okapi],
                  params: RecommendationParams) -> List[Tuple[Dict[str, Any], float]]:
        """
        Generates recommendations by performing semantic and keyword searches,
        then fusing the results using RRF.
        """
        if not resource_metadata:
            logger.warning("Resource metadata is empty. Cannot generate recommendations.")
            return []
        
        num_docs_in_corpus = len(resource_metadata)
        if resource_embeddings is not None and num_docs_in_corpus != resource_embeddings.shape[0]:
            logger.error(f"Mismatch between metadata count ({num_docs_in_corpus}) and embeddings count ({resource_embeddings.shape[0]}). Cannot proceed reliably.")
            # Depending on strictness, could return [] or raise error
            # For now, we'll try to proceed but semantic search might be unreliable or fail.

        logger.info(f"Starting recommendation for query: '{query[:100]}...'")
        logger.debug(f"Recommendation params: {params}")

        # Preprocess the query if query preprocessing is enabled
        processed_query = query
        if self.enable_query_preprocessing and self.query_processor:
            try:
                # --- MODIFICATION 2: Use the new parameter from the params object ---
                processed_query = self.query_processor.preprocess_query(
                    query, 
                    expand_synonyms=params.expand_synonyms
                )
                logger.info(f"Query preprocessing (synonyms {'enabled' if params.expand_synonyms else 'disabled'}): '{query}' -> '{processed_query}'")
            except Exception as e:
                logger.error(f"Query preprocessing failed: {e}. Using original query.", exc_info=True)
                processed_query = query
        else:
            logger.debug("Query preprocessing disabled or not available. Using original query.")

        query_embedding: Optional[np.ndarray] = None
        if self.embed_model and resource_embeddings is not None:
            try:
                # Use processed query for semantic search
                query_embedding = self.embed_model.encode(processed_query, task_type="RETRIEVAL_QUERY")
                logger.debug(f"Query embedding generated with shape: {query_embedding.shape if query_embedding is not None else 'None'}")
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
                query_embedding = None 
        elif not self.embed_model:
            logger.info("No embedding model provided. Skipping semantic search.")
        elif resource_embeddings is None:
            logger.info("Resource embeddings not available. Skipping semantic search.")


        # Perform semantic search
        semantic_results: List[Tuple[int, float]] = []
        if query_embedding is not None and resource_embeddings is not None:
            semantic_results = self._semantic_search(query_embedding, resource_embeddings, params.semantic_candidates)
            logger.debug(f"Semantic search returned {len(semantic_results)} results.")
        else:
            logger.debug("Skipping semantic search due to missing query embedding or resource embeddings.")

        # Perform keyword search using processed query
        keyword_results: List[Tuple[int, float]] = []
        if bm25_index is not None: # bm25_index can be None
            keyword_results = self._keyword_search(processed_query, bm25_index, num_docs_in_corpus, params.keyword_candidates)
            logger.debug(f"Keyword search returned {len(keyword_results)} results.")
        else:
            logger.debug("Skipping keyword search due to missing BM25 index.")

        # Fuse results
        # Ensure that RRF gets lists of (doc_index, score)
        # Filter out empty lists before passing to RRF
        lists_for_fusion = []
        if semantic_results:
            lists_for_fusion.append(semantic_results)
        if keyword_results:
            lists_for_fusion.append(keyword_results)
        
        if not lists_for_fusion:
            logger.info("No results from either semantic or keyword search to fuse.")
            return []

        fused_results_by_index = self._reciprocal_rank_fusion(lists_for_fusion, k_rrf=params.fusion_k)
        logger.debug(f"RRF returned {len(fused_results_by_index)} fused results.")

        # Sort fused results by score
        # items() gives list of (doc_index, score)
        sorted_fused_results = sorted(fused_results_by_index.items(), key=lambda item: item[1], reverse=True)

        # Map indices to metadata and prepare final recommendations
        final_recommendations: List[Tuple[Dict[str, Any], float]] = []
        processed_indices = set()

        for doc_index, score in sorted_fused_results:
            if doc_index in processed_indices:
                logger.debug(f"Skipping already processed document index: {doc_index}")
                continue
            processed_indices.add(doc_index)

            if 0 <= doc_index < num_docs_in_corpus:
                metadata_item = resource_metadata[doc_index]
                
                # CRITICAL CHECK: Ensure either 'entry_id' or 'arxiv_entry_id' key exists, or try to derive from URL
                entry_id = metadata_item.get('entry_id')
                arxiv_entry_id = metadata_item.get('arxiv_entry_id')
                doc_url = metadata_item.get('url')

                if not entry_id and not arxiv_entry_id:
                    if doc_url and isinstance(doc_url, str):
                        try:
                            # Example: 'http://arxiv.org/pdf/2410.12837v1' -> '2410.12837v1'
                            derived_id = doc_url.split('/')[-1]
                            if derived_id:
                                logger.warning(f"Document at index {doc_index} missing 'entry_id' and 'arxiv_entry_id'. Using derived ID '{derived_id}' from URL: {doc_url}")
                                metadata_item['entry_id'] = derived_id
                                entry_id = derived_id
                            else:
                                logger.error(f"Document at index {doc_index} in resource_metadata is missing 'entry_id', 'arxiv_entry_id', and could not derive a valid ID from URL '{doc_url}'. Metadata: {metadata_item}. Skipping this item.")
                                continue
                        except Exception as e:
                            logger.error(f"Error deriving ID from URL '{doc_url}' for document at index {doc_index}. Metadata: {metadata_item}. Error: {e}. Skipping this item.")
                            continue
                    else:
                        logger.error(f"Document at index {doc_index} in resource_metadata is missing 'entry_id', 'arxiv_entry_id', and has no valid 'url' to derive an ID. Metadata: {metadata_item}. Skipping this item.")
                        continue
                    
                # Use entry_id if available, otherwise fall back to arxiv_entry_id
                doc_id_for_log = entry_id or arxiv_entry_id or f"[No ID at index {doc_index}]"
                title_for_log = metadata_item.get('title', f"[No Title - ID: {doc_id_for_log}]")
                logger.debug(f"  Considering doc index {doc_index} (ID: {doc_id_for_log}, Title: \\\'{title_for_log}\\\') with RRF score {score:.4f}")
                final_recommendations.append((metadata_item, score))
            else:
                logger.warning(f"Document index {doc_index} from fused results is out of bounds for resource_metadata (len: {num_docs_in_corpus}). Skipping.")
            
            if len(final_recommendations) >= params.top_n_final:
                break
        
        logger.info(f"Returning {len(final_recommendations)} final recommendations (top_n_final={params.top_n_final}).")
        return final_recommendations[:params.top_n_final]