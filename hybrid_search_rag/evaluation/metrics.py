# hybrid_search_rag/evaluation/metrics.py
"""
This module provides functions to calculate common information retrieval (IR)
evaluation metrics. These metrics are essential for quantifying the performance
of search and recommendation systems.
"""
import numpy as np
import math
import random
from typing import List, Set, Union, Dict, Any

# --- Helper function for relevance checking ---
def _is_relevant(doc_id: Union[str, int], relevant_set: Set[Union[str, int]]) -> int:
    """Checks if a document ID is in the set of relevant documents."""
    return 1 if doc_id in relevant_set else 0

# --- Precision@K ---
def precision_at_k(
    retrieved_doc_ids: List[Union[str, int]],
    relevant_doc_ids: Set[Union[str, int]],
    k: int
) -> float:
    """
    Calculates Precision@K.
    Precision@K is the proportion of retrieved documents in the top K
    that are relevant.

    Formula:
    P@K = (Number of relevant documents in top K) / K

    Args:
        retrieved_doc_ids: An ordered list of document IDs retrieved by the system.
        relevant_doc_ids: A set of document IDs known to be relevant for the query.
        k: The number of top documents to consider.

    Returns:
        The Precision@K score, a float between 0.0 and 1.0.
    """
    if k <= 0:
        # ValueError is appropriate for invalid argument values.
        raise ValueError("k must be a positive integer.")
    if not retrieved_doc_ids: # If nothing is retrieved, precision is 0.
        return 0.0

    # Consider only the top k retrieved documents.
    top_k_retrieved = retrieved_doc_ids[:k]
    # Count how many of these top k documents are in the set of relevant documents.
    num_relevant_in_top_k = sum(_is_relevant(doc_id, relevant_doc_ids) for doc_id in top_k_retrieved)

    # Precision is the number of relevant items in top k divided by k.
    return num_relevant_in_top_k / k

# --- Recall@K ---
def recall_at_k(
    retrieved_doc_ids: List[Union[str, int]],
    relevant_doc_ids: Set[Union[str, int]],
    k: int
) -> float:
    """
    Calculates Recall@K.
    Recall@K is the proportion of all relevant documents that are
    found in the top K retrieved documents.

    Formula:
    R@K = (Number of relevant documents in top K) / (Total number of relevant documents)

    Args:
        retrieved_doc_ids: An ordered list of document IDs retrieved by the system.
        relevant_doc_ids: A set of document IDs known to be relevant for the query.
        k: The number of top documents to consider.

    Returns:
        The Recall@K score, a float between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    total_relevant_docs = len(relevant_doc_ids)
    if total_relevant_docs == 0:
        # If there are no relevant documents, recall is conventionally 0 (or undefined).
        # Returning 0.0 is a common practice. If all (zero) relevant docs are retrieved,
        # some might argue for 1.0, but 0.0 is safer if nothing was expected.
        return 0.0
    
    if not retrieved_doc_ids: # Nothing retrieved, so no relevant ones found.
        return 0.0

    top_k_retrieved = retrieved_doc_ids[:k]
    num_relevant_in_top_k = sum(_is_relevant(doc_id, relevant_doc_ids) for doc_id in top_k_retrieved)

    return num_relevant_in_top_k / total_relevant_docs


# --- Average Precision (AP) ---
def average_precision(
    retrieved_doc_ids: List[Union[str, int]],
    relevant_doc_ids: Set[Union[str, int]]
) -> float:
    """
    Calculates Average Precision (AP) for a single query.
    AP averages the precision@k values calculated at the rank of each
    relevant document retrieved. It rewards systems that retrieve relevant
    documents earlier in the ranking.

    Formula:
    AP = sum_{i=1}^{N} (P@i * rel(i)) / (Total number of relevant documents)
    where:
        N is the total number of retrieved documents.
        P@i is Precision at rank i.
        rel(i) is 1 if the document at rank i is relevant, 0 otherwise.

    Args:
        retrieved_doc_ids: An ordered list of document IDs retrieved by the system.
        relevant_doc_ids: A set of document IDs known to be relevant for the query.

    Returns:
        The Average Precision score for the query, a float between 0.0 and 1.0.
    """
    total_relevant_docs = len(relevant_doc_ids)
    if total_relevant_docs == 0:
        return 0.0 # No relevant documents, AP is 0.
    if not retrieved_doc_ids:
        return 0.0 # Nothing retrieved, AP is 0.

    hits = 0 # Number of relevant documents found so far.
    sum_precisions = 0.0
    # Iterate through the retrieved documents and their ranks.
    for i, doc_id in enumerate(retrieved_doc_ids):
        if _is_relevant(doc_id, relevant_doc_ids):
            hits += 1
            # Precision at current rank (i+1 because ranks are 1-based).
            precision_at_this_rank = hits / (i + 1)
            sum_precisions += precision_at_this_rank

    return sum_precisions / total_relevant_docs

# --- Mean Average Precision (MAP) ---
def mean_average_precision(list_of_aps: List[float]) -> float:
    """
    Calculates Mean Average Precision (MAP).
    MAP is the mean of Average Precision (AP) scores over a set of queries.

    Formula:
    MAP = (1/|Q|) * sum_{q in Q} AP(q)
    where:
        Q is the set of queries.
        AP(q) is the Average Precision for query q.

    Args:
        list_of_aps: A list of Average Precision scores, one for each query.

    Returns:
        The Mean Average Precision score, a float between 0.0 and 1.0.
    """
    if not list_of_aps: # If the list is empty, MAP is 0.
        return 0.0
    # np.mean returns a numpy float type, explicitly cast to Python float for consistency.
    return float(np.mean(list_of_aps))

# --- Discounted Cumulative Gain (DCG@K) ---
def dcg_at_k(
    # retrieved_doc_ids: List[Union[str, int]], # Not strictly needed if relevance_scores are already for top K
    relevance_scores_for_retrieved_top_k: List[float], # Relevance scores for the actual top K retrieved docs
    k: int, # This k should match the length of relevance_scores_for_retrieved_top_k
    log_base: float = 2.0
) -> float:
    """
    Calculates Discounted Cumulative Gain (DCG)@K.
    DCG measures the usefulness (gain) of documents based on their position.
    Gain is discounted at lower ranks.

    Formula (common version for 0-indexed scores):
    DCG@K = sum_{i=0}^{K-1} (relevance_scores[i] / log_base(i+2))
    (i.e., rel_1/log(1+1) + rel_2/log(2+1) + ... for 1-indexed relevances)

    Args:
        relevance_scores_for_retrieved_top_k: A list of true relevance scores
                                              for the documents retrieved in the top K positions.
                                              Length of this list should be K (or min(K, num_retrieved)).
        k: The number of top documents considered (should match len of relevance_scores_for_retrieved_top_k).
        log_base: The base of the logarithm for discounting. Usually 2.

    Returns:
        The DCG@K score.
    """
    # k represents the number of scores in relevance_scores_for_retrieved_top_k.
    # If k is 0 (i.e., the list is empty), DCG is 0.
    if k == 0:
        return 0.0
    # k should not be negative if it's derived from len().
    if k < 0:
        raise ValueError("k cannot be negative.")
    
    # The relevance_scores_for_retrieved_top_k should already be sliced to k or fewer items.
    # k_actual = min(k, len(relevance_scores_for_retrieved_top_k)) # k_actual is just len(...)
    
    dcg = 0.0
    for i in range(len(relevance_scores_for_retrieved_top_k)):
        gain = relevance_scores_for_retrieved_top_k[i]
        # Discount factor: log_base(rank + 1). For 0-indexed i, rank is i+1.
        # So, denominator is log_base(i + 1 + 1) = log_base(i + 2) for the formula rel_i / log(i+1)
        # Or, if using rel_i / log(rank), then for rank = i+1, it's log_base(i+1).
        # Let's stick to a common formulation: sum rel_i / log2(i+1) where i is 1-based rank.
        # For 0-indexed loop variable `i`, the rank is `i+1`.
        # The discount is applied to ranks 2 onwards more heavily in some formulations (e.g. rel_1 + sum(rel_i/log(i)))
        # Using: sum_{j=1}^{K} (rel_j / log_base(j+1)) -> sum_{i=0}^{K-1} (rel_scores[i] / log_base(i+2))
        # Using: sum_{j=1}^{K} (rel_j / log_base(j)) if first element not discounted, or rel_1 + sum_{j=2}^{K} (rel_j / log_base(j))
        # A common one is: relevance_scores[i] / math.log(i + 2, log_base) for 0-indexed i
        # Rank 1 (i=0) -> discount log(2)
        # Rank 2 (i=1) -> discount log(3)
        discount = math.log(i + 2, log_base) 
        dcg += gain / discount
    return dcg

# --- Ideal Discounted Cumulative Gain (IDCG@K) ---
def idcg_at_k(
    all_true_relevance_scores_sorted_desc: List[float], # ALL true relevance scores for the query, sorted
    k: int,
    log_base: float = 2.0
) -> float:
    """
    Calculates Ideal Discounted Cumulative Gain (IDCG)@K.
    IDCG is the DCG score of a hypothetical ideal ranking.

    Args:
        all_true_relevance_scores_sorted_desc: A list of ALL known true relevance scores
                                               for a query, sorted in descending order.
        k: The number of top documents to consider for the ideal ranking.
        log_base: The base of the logarithm for discounting.

    Returns:
        The IDCG@K score.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Take the top K scores from the sorted list of all true relevances for the ideal ranking.
    ideal_top_k_relevances = all_true_relevance_scores_sorted_desc[:k]
    if not ideal_top_k_relevances: # No relevant items or k is effectively 0 for this list
        return 0.0

    # Calculate DCG for this ideal list.
    # This reuses the dcg_at_k logic by passing the ideal relevances.
    return dcg_at_k(ideal_top_k_relevances, len(ideal_top_k_relevances), log_base)


# --- Normalized Discounted Cumulative Gain (nDCG@K) ---
def ndcg_at_k(
    retrieved_doc_ids: List[Union[str, int]],
    true_relevance_map: Dict[Union[str, int], float], # Maps ALL doc_ids (retrieved or not) to true relevance
    k: int,
    log_base: float = 2.0
) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (nDCG)@K.
    nDCG = DCG@K / IDCG@K. Score is between 0.0 and 1.0.

    Args:
        retrieved_doc_ids: An ordered list of document IDs retrieved by the system.
        true_relevance_map: A dictionary mapping document IDs to their true relevance scores.
                            This map should ideally contain all documents with known relevance for the query.
        k: The number of top documents to consider.
        log_base: The base of the logarithm for discounting.

    Returns:
        The nDCG@K score.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # 1. Get relevance scores for the actual retrieved top-K items
    # Slice retrieved_doc_ids to top K first, then get their scores.
    actual_top_k_retrieved_ids = retrieved_doc_ids[:k]
    retrieved_relevance_scores_for_top_k = [true_relevance_map.get(doc_id, 0.0) for doc_id in actual_top_k_retrieved_ids]
    
    # Calculate DCG for the actual retrieved list (up to K items)
    current_dcg = dcg_at_k(retrieved_relevance_scores_for_top_k, len(retrieved_relevance_scores_for_top_k), log_base)

    # 2. Calculate IDCG
    # Get all true relevance scores from the map and sort them for IDCG
    if not true_relevance_map: # No relevance information available at all
        ideal_dcg = 0.0
    else:
        all_true_scores_sorted_desc = sorted(true_relevance_map.values(), reverse=True)
        ideal_dcg = idcg_at_k(all_true_scores_sorted_desc, k, log_base)

    if ideal_dcg == 0:
        # If IDCG is 0 (e.g., no relevant documents for the query, or k=0),
        # nDCG is 0, unless DCG is also 0 (then it could be 1 by some conventions,
        # but 0 is safer if there's nothing to gain).
        # If all true relevances are 0, then any retrieved item will also have 0 relevance, so DCG will be 0.
        return 0.0 
    
    return current_dcg / ideal_dcg


# --- Additional Advanced Metrics ---

# === EMBEDDING QUALITY METRICS ===

def embedding_similarity_distribution(
    query_embeddings: List[np.ndarray],
    doc_embeddings: List[np.ndarray],
    labels: Union[List[int], np.ndarray] = None
) -> Dict[str, float]:
    """
    Analyzes the distribution of cosine similarities between query and document embeddings.
    Useful for understanding embedding space quality and separation between relevant/irrelevant docs.
    
    Args:
        query_embeddings: List of query embedding vectors
        doc_embeddings: List of document embedding vectors  
        labels: Optional list of relevance labels (1 for relevant, 0 for irrelevant)
    
    Returns:
        Dictionary with similarity statistics
    """
    if len(query_embeddings) != len(doc_embeddings):
        raise ValueError("Query and document embedding lists must have same length")
    
    similarities = []
    for q_emb, d_emb in zip(query_embeddings, doc_embeddings):
        # Normalize embeddings for cosine similarity
        q_norm = q_emb / np.linalg.norm(q_emb)
        d_norm = d_emb / np.linalg.norm(d_emb)
        sim = np.dot(q_norm, d_norm)
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    stats = {
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'median_similarity': float(np.median(similarities))
    }
    
    if labels is not None:
        labels = np.array(labels)
        relevant_sims = similarities[labels == 1]
        irrelevant_sims = similarities[labels == 0]
        
        if len(relevant_sims) > 0:
            stats['mean_relevant_similarity'] = float(np.mean(relevant_sims))
        if len(irrelevant_sims) > 0:
            stats['mean_irrelevant_similarity'] = float(np.mean(irrelevant_sims))
        if len(relevant_sims) > 0 and len(irrelevant_sims) > 0:
            stats['similarity_separation'] = stats['mean_relevant_similarity'] - stats['mean_irrelevant_similarity']
    
    return stats


def embedding_space_coverage(
    embeddings: List[np.ndarray],
    sample_size: int = 1000
) -> Dict[str, float]:
    """
    Measures how well embeddings cover the semantic space.
    
    Args:
        embeddings: List of embedding vectors
        sample_size: Number of random pairs to sample for analysis
        
    Returns:
        Dictionary with coverage metrics
    """
    if len(embeddings) < 2:
        return {'mean_pairwise_distance': 0.0, 'std_pairwise_distance': 0.0}
    
    # Convert to numpy array for easier manipulation
    emb_array = np.array(embeddings)
    
    # Sample random pairs to avoid O(n²) computation
    n_embeddings = len(embeddings)
    n_pairs = min(sample_size, (n_embeddings * (n_embeddings - 1)) // 2)
    
    distances = []
    for _ in range(n_pairs):
        i, j = random.sample(range(n_embeddings), 2)
        # Normalize for cosine distance
        emb_i = emb_array[i] / np.linalg.norm(emb_array[i])
        emb_j = emb_array[j] / np.linalg.norm(emb_array[j])
        # Cosine distance = 1 - cosine similarity
        distance = 1 - np.dot(emb_i, emb_j)
        distances.append(distance)
    
    distances = np.array(distances)
    
    return {
        'mean_pairwise_distance': float(np.mean(distances)),
        'std_pairwise_distance': float(np.std(distances)),
        'min_pairwise_distance': float(np.min(distances)),
        'max_pairwise_distance': float(np.max(distances))
    }


# === RANKING QUALITY METRICS ===

def rank_correlation(
    predicted_ranks: List[int],
    true_ranks: List[int]
) -> float:
    """
    Calculates Spearman's rank correlation between predicted and true rankings.
    
    Args:
        predicted_ranks: List of predicted rank positions
        true_ranks: List of true rank positions
        
    Returns:
        Spearman correlation coefficient (-1 to 1)
    """
    if len(predicted_ranks) != len(true_ranks):
        raise ValueError("Predicted and true rank lists must have same length")
    
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(predicted_ranks, true_ranks)
    return float(correlation) if not np.isnan(correlation) else 0.0


def reciprocal_rank(
    retrieved_doc_ids: List[Union[str, int]],
    relevant_doc_ids: Set[Union[str, int]]
) -> float:
    """
    Calculates the reciprocal rank of the first relevant document.
    
    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs
        relevant_doc_ids: Set of relevant document IDs
        
    Returns:
        Reciprocal rank (1/rank of first relevant doc, 0 if none found)
    """
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)  # i is 0-indexed, rank is 1-indexed
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: List[float]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR) from a list of reciprocal ranks.
    
    Args:
        reciprocal_ranks: List of reciprocal rank values
        
    Returns:
        Mean reciprocal rank
    """
    if not reciprocal_ranks:
        return 0.0
    return float(np.mean(reciprocal_ranks))


# === RESPONSE GENERATION QUALITY METRICS ===

def bleu_score(reference: str, candidate: str, n_gram: int = 4) -> float:
    """
    Calculates BLEU score for response generation quality.
    Simplified implementation for basic n-gram overlap.
    
    Args:
        reference: Reference (ground truth) text
        candidate: Generated candidate text
        n_gram: Maximum n-gram size to consider
        
    Returns:
        BLEU score (0 to 1)
    """
    def get_ngrams(text: str, n: int) -> Set[str]:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))
    
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if not cand_words:
        return 0.0
    
    scores = []
    for n in range(1, min(n_gram + 1, len(cand_words) + 1)):
        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)
        
        if not cand_ngrams:
            scores.append(0.0)
            continue
            
        overlap = len(ref_ngrams.intersection(cand_ngrams))
        precision = overlap / len(cand_ngrams)
        scores.append(precision)
    
    if not scores:
        return 0.0
    
    # Geometric mean of n-gram precisions
    return float(np.exp(np.mean(np.log(np.array(scores) + 1e-10))))


def rouge_l_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculates ROUGE-L score based on longest common subsequence.
    
    Args:
        reference: Reference (ground truth) text
        candidate: Generated candidate text
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    def lcs_length(s1: List[str], s2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if not ref_words or not cand_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    lcs_len = lcs_length(ref_words, cand_words)
    
    precision = lcs_len / len(cand_words)
    recall = lcs_len / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


# === HYBRID SEARCH SPECIFIC METRICS ===

def fusion_effectiveness(
    dense_results: List[Union[str, int]],
    sparse_results: List[Union[str, int]], 
    fused_results: List[Union[str, int]],
    relevant_doc_ids: Set[Union[str, int]],
    k: int = 10
) -> Dict[str, float]:
    """
    Measures how effectively the fusion algorithm combines dense and sparse retrieval.
    
    Args:
        dense_results: Results from dense/semantic retrieval
        sparse_results: Results from sparse/keyword retrieval  
        fused_results: Results from fusion algorithm
        relevant_doc_ids: Set of relevant document IDs
        k: Number of top results to evaluate
        
    Returns:
        Dictionary comparing effectiveness of different approaches
    """
    dense_p_at_k = precision_at_k(dense_results, relevant_doc_ids, k)
    sparse_p_at_k = precision_at_k(sparse_results, relevant_doc_ids, k) 
    fused_p_at_k = precision_at_k(fused_results, relevant_doc_ids, k)
    
    dense_r_at_k = recall_at_k(dense_results, relevant_doc_ids, k)
    sparse_r_at_k = recall_at_k(sparse_results, relevant_doc_ids, k)
    fused_r_at_k = recall_at_k(fused_results, relevant_doc_ids, k)
    
    return {
        'dense_precision_at_k': dense_p_at_k,
        'sparse_precision_at_k': sparse_p_at_k,
        'fused_precision_at_k': fused_p_at_k,
        'precision_improvement_over_dense': fused_p_at_k - dense_p_at_k,
        'precision_improvement_over_sparse': fused_p_at_k - sparse_p_at_k,
        
        'dense_recall_at_k': dense_r_at_k,
        'sparse_recall_at_k': sparse_r_at_k, 
        'fused_recall_at_k': fused_r_at_k,
        'recall_improvement_over_dense': fused_r_at_k - dense_r_at_k,
        'recall_improvement_over_sparse': fused_r_at_k - sparse_r_at_k,
        
        'fusion_effectiveness_score': (fused_p_at_k + fused_r_at_k) / 2
    }


def retrieval_diversity(
    retrieved_doc_ids: List[Union[str, int]],
    doc_embeddings_map: Dict[Union[str, int], np.ndarray],
    k: int = 10
) -> float:
    """
    Measures diversity of retrieved results based on embedding similarity.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        doc_embeddings_map: Map from document ID to embedding vector
        k: Number of top results to evaluate
        
    Returns:
        Diversity score (higher = more diverse)
    """
    top_k_ids = retrieved_doc_ids[:k]
    
    if len(top_k_ids) < 2:
        return 0.0
    
    # Get embeddings for top-k documents
    embeddings = []
    for doc_id in top_k_ids:
        if doc_id in doc_embeddings_map:
            embeddings.append(doc_embeddings_map[doc_id])
    
    if len(embeddings) < 2:
        return 0.0
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            emb_i = embeddings[i] / np.linalg.norm(embeddings[i])
            emb_j = embeddings[j] / np.linalg.norm(embeddings[j])
            sim = np.dot(emb_i, emb_j)
            similarities.append(sim)
    
    # Diversity is inverse of average similarity
    avg_similarity = np.mean(similarities)
    return float(1.0 - avg_similarity)


# === COMPREHENSIVE EVALUATION METRICS CLASS ===

class ComprehensiveMetrics:
    """
    A comprehensive metrics calculator that computes all evaluation metrics.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_ir_metrics(
        self,
        retrieved_doc_ids: List[Union[str, int]],
        relevant_doc_ids: Set[Union[str, int]],
        relevance_scores_map: Dict[Union[str, int], float] = None,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """Calculate all information retrieval metrics."""
        
        results = {}
        
        # Basic metrics for each k
        for k in k_values:
            results[f'precision_at_{k}'] = precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            results[f'recall_at_{k}'] = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            
            if relevance_scores_map:
                results[f'ndcg_at_{k}'] = ndcg_at_k(retrieved_doc_ids, relevance_scores_map, k)
        
        # Single-value metrics
        results['average_precision'] = average_precision(retrieved_doc_ids, relevant_doc_ids)
        results['reciprocal_rank'] = reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)
        
        return results
    
    def calculate_embedding_metrics(
        self,
        query_embeddings: List[np.ndarray],
        doc_embeddings: List[np.ndarray], 
        relevance_labels: List[int] = None
    ) -> Dict[str, Any]:
        """Calculate embedding quality metrics."""
        
        results = {}
        
        # Similarity distribution analysis
        sim_stats = embedding_similarity_distribution(
            query_embeddings, doc_embeddings, relevance_labels
        )
        results.update({f'embedding_{k}': v for k, v in sim_stats.items()})
        
        # Embedding space coverage
        coverage_stats = embedding_space_coverage(doc_embeddings)
        results.update({f'coverage_{k}': v for k, v in coverage_stats.items()})
        
        return results
    
    def calculate_response_quality_metrics(
        self,
        reference_texts: List[str],
        generated_texts: List[str]
    ) -> Dict[str, Any]:
        """Calculate response generation quality metrics."""
        
        if len(reference_texts) != len(generated_texts):
            raise ValueError("Reference and generated text lists must have same length")
        
        bleu_scores = []
        rouge_scores = {'precision': [], 'recall': [], 'f1': []}
        
        for ref, gen in zip(reference_texts, generated_texts):
            bleu_scores.append(bleu_score(ref, gen))
            rouge = rouge_l_score(ref, gen)
            rouge_scores['precision'].append(rouge['precision'])
            rouge_scores['recall'].append(rouge['recall'])
            rouge_scores['f1'].append(rouge['f1'])
        
        return {
            'mean_bleu_score': float(np.mean(bleu_scores)),
            'mean_rouge_precision': float(np.mean(rouge_scores['precision'])),
            'mean_rouge_recall': float(np.mean(rouge_scores['recall'])),
            'mean_rouge_f1': float(np.mean(rouge_scores['f1']))
        }


# === EVALUATION REPORTING ===

def generate_metrics_report(metrics_dict: Dict[str, Any]) -> str:
    """
    Generates a formatted metrics report.
    
    Args:
        metrics_dict: Dictionary containing all calculated metrics
        
    Returns:
        Formatted string report
    """
    report = ["=" * 60]
    report.append("COMPREHENSIVE RETRIEVAL EVALUATION REPORT")
    report.append("=" * 60)
    
    # Information Retrieval Metrics
    ir_metrics = {k: v for k, v in metrics_dict.items() 
                  if any(metric in k for metric in ['precision', 'recall', 'ndcg', 'average_precision', 'reciprocal_rank'])}
    
    if ir_metrics:
        report.append("\n📊 INFORMATION RETRIEVAL METRICS")
        report.append("-" * 40)
        for metric, value in sorted(ir_metrics.items()):
            if isinstance(value, float):
                report.append(f"{metric:.<30} {value:.4f}")
            else:
                report.append(f"{metric:.<30} {value}")
    
    # Embedding Quality Metrics  
    embedding_metrics = {k: v for k, v in metrics_dict.items() if 'embedding' in k or 'coverage' in k}
    
    if embedding_metrics:
        report.append("\n🎯 EMBEDDING QUALITY METRICS")
        report.append("-" * 40)
        for metric, value in sorted(embedding_metrics.items()):
            if isinstance(value, float):
                report.append(f"{metric:.<30} {value:.4f}")
            else:
                report.append(f"{metric:.<30} {value}")
    
    # Response Quality Metrics
    response_metrics = {k: v for k, v in metrics_dict.items() 
                       if any(metric in k for metric in ['bleu', 'rouge'])}
    
    if response_metrics:
        report.append("\n📝 RESPONSE GENERATION METRICS")
        report.append("-" * 40)
        for metric, value in sorted(response_metrics.items()):
            if isinstance(value, float):
                report.append(f"{metric:.<30} {value:.4f}")
            else:
                report.append(f"{metric:.<30} {value}")
    
    # Fusion Effectiveness Metrics
    fusion_metrics = {k: v for k, v in metrics_dict.items() if 'fusion' in k or 'improvement' in k}
    
    if fusion_metrics:
        report.append("\n🔄 HYBRID FUSION METRICS") 
        report.append("-" * 40)
        for metric, value in sorted(fusion_metrics.items()):
            if isinstance(value, float):
                report.append(f"{metric:.<30} {value:.4f}")
            else:
                report.append(f"{metric:.<30} {value}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)

# Update the example usage section
if __name__ == '__main__':
    # Extended example usage with comprehensive metrics
    print("Testing Comprehensive Metrics System...")
    
    # Example data
    retrieved_ids = ["docA", "docB", "docC", "docD", "docE", "docF", "docG"]
    relevant_set = {"docA", "docC", "docE", "docH", "docI"}
    relevance_scores = {
        "docA": 3.0, "docB": 1.0, "docC": 2.0, "docD": 0.0, "docE": 3.0,
        "docF": 0.0, "docG": 1.0, "docH": 3.0, "docI": 2.0, "docJ": 0.0
    }
    
    # Initialize comprehensive metrics calculator
    comp_metrics = ComprehensiveMetrics()
    
    # Calculate IR metrics
    ir_results = comp_metrics.calculate_all_ir_metrics(
        retrieved_ids, relevant_set, relevance_scores, [5, 10, 20]
    )
    
    # Example embedding metrics (random data for demo)
    query_embs = [np.random.randn(384) for _ in range(5)]
    doc_embs = [np.random.randn(384) for _ in range(5)] 
    labels = [1, 0, 1, 0, 1]
    
    embedding_results = comp_metrics.calculate_embedding_metrics(query_embs, doc_embs, labels)
    
    # Combine all results
    all_metrics = {**ir_results, **embedding_results}
    
    # Generate and print report
    report = generate_metrics_report(all_metrics)
    print(report)
