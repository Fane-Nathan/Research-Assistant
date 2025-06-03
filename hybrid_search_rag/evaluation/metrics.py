# hybrid_search_rag/evaluation/metrics.py
"""
This module provides functions to calculate common information retrieval (IR)
evaluation metrics. These metrics are essential for quantifying the performance
of search and recommendation systems.
"""
import numpy as np
import math
from typing import List, Set, Union, Dict

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


# --- Example Usage (for testing purposes) ---
if __name__ == '__main__':
    # Example data for a single query
    retrieved_ids_example: List[Union[str,int]] = ["docA", "docB", "docC", "docD", "docE", "docF", "docG"]
    relevant_set_example: Set[Union[str,int]] = {"docA", "docC", "docE", "docH", "docI"} 

    true_relevances_example: Dict[Union[str,int], float] = {
        "docA": 3.0, "docB": 1.0, "docC": 2.0, "docD": 0.0, "docE": 3.0,
        "docF": 0.0, "docG": 1.0, "docH": 3.0, "docI": 2.0, "docJ": 0.0 
    }
    k_val_example = 5

    print(f"--- Metrics for K={k_val_example} ---")

    p_at_k = precision_at_k(retrieved_ids_example, relevant_set_example, k_val_example)
    print(f"Precision@{k_val_example}: {p_at_k:.4f}") # Expected: 3/5 = 0.6

    r_at_k = recall_at_k(retrieved_ids_example, relevant_set_example, k_val_example)
    print(f"Recall@{k_val_example}: {r_at_k:.4f}") # Expected: 3/5 = 0.6

    ap_score = average_precision(retrieved_ids_example, relevant_set_example)
    print(f"Average Precision (AP): {ap_score:.4f}") # Expected: (1/1 * 1 + 2/3 * 1 + 3/5 * 1) / 5 = (1 + 0.6667 + 0.6) / 5 = 2.2667 / 5 = 0.4533

    # For nDCG@K
    # Scores for retrieved top 5: [A:3, B:1, C:2, D:0, E:3]
    retrieved_rels_for_ndcg_example = [true_relevances_example.get(doc_id, 0.0) for doc_id in retrieved_ids_example[:k_val_example]]
    
    dcg_val = dcg_at_k(retrieved_rels_for_ndcg_example, k_val_example) # k_val_example matches length
    print(f"DCG@{k_val_example}: {dcg_val:.4f}") 
    # DCG@5 for [3,1,2,0,3]
    # 3/log2(0+2) + 1/log2(1+2) + 2/log2(2+2) + 0/log2(3+2) + 3/log2(4+2)
    # 3/log2(2) + 1/log2(3) + 2/log2(4) + 0/log2(5) + 3/log2(6)
    # 3/1 + 1/1.58496 + 2/2 + 0/2.3219 + 3/2.58496
    # 3 + 0.6309 + 1 + 0 + 1.1605 = 5.7914

    all_true_scores_sorted_desc_example = sorted(true_relevances_example.values(), reverse=True)
    idcg_val = idcg_at_k(all_true_scores_sorted_desc_example, k_val_example)
    print(f"IDCG@{k_val_example}: {idcg_val:.4f}")
    # Ideal top 5 relevances: [3.0, 3.0, 3.0, 2.0, 2.0]
    # IDCG@5 for [3,3,3,2,2]
    # 3/log2(2) + 3/log2(3) + 3/log2(4) + 2/log2(5) + 2/log2(6)
    # 3 + 1.8928 + 1.5 + 0.8613 + 0.7737 = 8.0278

    ndcg_val = ndcg_at_k(retrieved_ids_example, true_relevances_example, k_val_example)
    print(f"nDCG@{k_val_example}: {ndcg_val:.4f}") # Expected: 5.7914 / 8.0278 = 0.7214

    # Test MAP
    map_score = mean_average_precision([ap_score, 0.8, 0.6])
    print(f"MAP score: {map_score:.4f}") # (0.4533 + 0.8 + 0.6) / 3 = 1.8533 / 3 = 0.6178
