"""
Query Processor for StudyAssistant Project

This module provides functionalities for preprocessing and expanding user queries
to improve retrieval accuracy and relevance in the RAG system.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import re

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    # Ensure necessary NLTK data is downloaded (optional, can be handled by user)
    # nltk.download('wordnet', quiet=True)
    # nltk.download('punkt', quiet=True)
    # nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None
    wordnet = None
    word_tokenize = None


logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Handles query preprocessing tasks such as cleaning, normalization,
    and expansion.
    """

    def __init__(self, use_nltk_expansion: bool = True):
        """
        Initializes the QueryProcessor.

        Args:
            use_nltk_expansion (bool): Whether to attempt query expansion using NLTK/WordNet.
                                       Defaults to True. NLTK must be installed.
        """
        self.use_nltk_expansion = use_nltk_expansion and NLTK_AVAILABLE
        if use_nltk_expansion and not NLTK_AVAILABLE:
            logger.warning(
                "NLTK is not available or WordNet/Punkt data is missing. "
                "Query expansion with WordNet will be disabled. "
                "Please install NLTK and download necessary resources: "
                "`pip install nltk` then run `python -m nltk.downloader wordnet punkt averaged_perceptron_tagger`"
            )
        elif self.use_nltk_expansion:
            logger.info("QueryProcessor initialized with NLTK-based synonym expansion enabled.")
        else:
            logger.info("QueryProcessor initialized without NLTK-based synonym expansion.")


    def _clean_query(self, query: str) -> str:
        """
        Basic cleaning of the query string.

        Args:
            query (str): The input query.

        Returns:
            str: The cleaned query.
        """
        query = query.strip()
        query = re.sub(r'\s+', ' ', query) 
        return query

    def _get_synonyms_nltk(self, word: str, pos_tag: Optional[str] = None) -> List[str]:
        """
        Gets synonyms for a word using NLTK WordNet.

        Args:
            word (str): The word to find synonyms for.
            pos_tag (str, optional): Part-of-speech tag for the word.

        Returns:
            List[str]: A list of synonyms.
        """
        if not self.use_nltk_expansion or not wordnet:
            return []

        synonyms = set()
        wn_pos = None
        if pos_tag:
            if pos_tag.startswith('N'):
                wn_pos = wordnet.NOUN
            elif pos_tag.startswith('V'):
                wn_pos = wordnet.VERB
            elif pos_tag.startswith('J'):
                wn_pos = wordnet.ADJ
            elif pos_tag.startswith('R'):
                wn_pos = wordnet.ADV
        
        for syn in wordnet.synsets(word, pos=wn_pos):
            if syn is not None and hasattr(syn, "lemmas"):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
        return list(synonyms)[:3]

    def expand_query_with_synonyms(self, query: str) -> str:
        """
        Expands the query by adding synonyms for key terms using NLTK.

        Args:
            query (str): The original query.

        Returns:
            str: The expanded query.
        """
        if not self.use_nltk_expansion or not word_tokenize or not nltk:
            logger.debug("NLTK-based synonym expansion skipped (NLTK not available or disabled).")
            return query

        tokens = word_tokenize(query)
        tagged_tokens = nltk.pos_tag(tokens)
        
        expanded_terms = []
        original_query_terms = query.lower().split()

        for word, tag in tagged_tokens:
            # Avoid expanding very common words or if word is too short
            if len(word) > 2 and word.lower() not in nltk.corpus.stopwords.words('english'):
                synonyms = self._get_synonyms_nltk(word, tag)
                if synonyms:
                    expanded_terms.extend([s for s in synonyms if s.lower() not in original_query_terms and s.lower() not in expanded_terms])
        
        if expanded_terms:
            expansion_str = " ".join(expanded_terms)
            expanded_query = f"{query} {expansion_str}"
            logger.debug(f"Original query: '{query}'. Expanded query: '{expanded_query}'")
            return expanded_query
        
        logger.debug(f"No suitable synonyms found for query: '{query}'")
        return query

    def preprocess_query(self, query: str, expand_synonyms: bool = True) -> str:
        """
        Applies a series of preprocessing steps to the query.

        Args:
            query (str): The raw user query.
            expand_synonyms (bool): Whether to perform synonym expansion. Defaults to True.

        Returns:
            str: The processed query.
        """
        logger.debug(f"Original query for preprocessing: '{query}'")
        
        cleaned_query = self._clean_query(query)
        logger.debug(f"Cleaned query: '{cleaned_query}'")
        
        processed_query = cleaned_query
        if expand_synonyms and self.use_nltk_expansion:
            processed_query = self.expand_query_with_synonyms(cleaned_query)
            logger.debug(f"Query after synonym expansion: '{processed_query}'")
        
        # Future enhancements:
        # - Stop word removal (carefully, can hurt keyword search)
        # - Stemming/Lemmatization (can also be risky, might over-generalize)
        # - Query segmentation for complex queries
        # - Named Entity Recognition to identify key entities

        logger.info(f"Final processed query: '{processed_query}'")
        return processed_query

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    if NLTK_AVAILABLE and nltk is not None:
        try:
            nltk.data.find('corpora/wordnet.zip')
            nltk.data.find('tokenizers/punkt.zip')
            nltk.data.find('taggers/averaged_perceptron_tagger.zip')
            nltk.data.find('corpora/stopwords.zip')
        except Exception as e:
            logger.error(f"NLTK resources missing for example: {e}")
            logger.error("Please run: python -m nltk.downloader wordnet punkt averaged_perceptron_tagger stopwords")
            NLTK_AVAILABLE = False

    qp = QueryProcessor(use_nltk_expansion=NLTK_AVAILABLE)
    
    test_queries = [
        "recent advancements in machine learning",
        "   benefits of   artificial intelligence research  ",
        "applications of deep learning in healthcare",
        "challenges in natural language processing",
        "ethical considerations for AI systems"
    ]
    
    for t_query in test_queries:
        print(f"\nOriginal: {t_query}")
        processed = qp.preprocess_query(t_query)
        print(f"Processed: {processed}")

    print("\n--- Testing without NLTK expansion ---")
    qp_no_nltk = QueryProcessor(use_nltk_expansion=False)
    for t_query in test_queries:
        print(f"\nOriginal: {t_query}")
        processed = qp_no_nltk.preprocess_query(t_query, expand_synonyms=False)
        print(f"Processed (no NLTK): {processed}")
