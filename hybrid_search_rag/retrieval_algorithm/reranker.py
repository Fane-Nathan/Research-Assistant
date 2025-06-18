# hybrid_search_rag/retrieval_algorithm/reranker.py
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)

class ReRanker:
    """
    A class to re-rank a list of documents based on a query using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the ReRanker by loading a pre-trained Cross-Encoder model.
        
        Args:
            model_name (str): The name of the cross-encoder model to use from Hugging Face.
        """
        try:
            logger.info(f"Initializing ReRanker with model: {model_name}")
            self.model = CrossEncoder(model_name)
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model '{model_name}'. Please ensure 'sentence-transformers' is installed and the model name is correct. Error: {e}", exc_info=True)
            self.model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Re-ranks a list of documents based on their relevance to a given query.

        Args:
            query (str): The user query.
            documents (List[Dict[str, Any]]): The list of candidate documents from the initial retrieval stage. 
                                             Each document is expected to have a 'text' key.

        Returns:
            List[Tuple[Dict[str, Any], float]]: A sorted list of (document, score) tuples, ranked by the new relevance score.
        """
        if not self.model:
            logger.error("ReRanker model is not initialized. Cannot perform re-ranking.")
            return [(doc, 0.0) for doc in documents]

        if not documents:
            return []

        sentence_pairs = [(query, doc.get('text', '')) for doc in documents]
        
        logger.debug(f"Re-ranking {len(documents)} documents for query: '{query[:100]}...'")
        
        scores = self.model.predict(sentence_pairs)
        
        reranked_results = list(zip(documents, scores))
        
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug("Re-ranking complete.")
        return reranked_results