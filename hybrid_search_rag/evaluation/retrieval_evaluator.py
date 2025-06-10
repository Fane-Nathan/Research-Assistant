import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

# All necessary imports from OTHER files are here.
# The incorrect self-import has been removed.
from ..retrieval_algorithm.hybrid_recommender import (
    HybridRecommender,
    RecommendationParams,
)
from ..embedding_services.gemini_embedder import EmbeddingModel

class RetrievalEvaluationMetrics:
    """A class to hold the evaluation metrics for the retrieval algorithm."""
    def __init__(self):
        self.hit_rates: List[float] = []
        self.mrrs: List[float] = []
        self.ndcgs: List[float] = []
        self.precisions: List[float] = []
        self.recalls: List[float] = []
        self.f1_scores: List[float] = []

    def update_ranking_metrics(self, hit_rate: float, mrr: float, ndcg: float):
        self.hit_rates.append(hit_rate)
        self.mrrs.append(mrr)
        self.ndcgs.append(ndcg)

    def update_classification_metrics(self, precision: float, recall: float, f1_score: float):
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1_score)

    def get_average_metrics(self) -> Dict[str, float]:
        return {
            "hit_rate": float(np.mean(self.hit_rates)) if self.hit_rates else 0.0,
            "mrr": float(np.mean(self.mrrs)) if self.mrrs else 0.0,
            "ndcg": float(np.mean(self.ndcgs)) if self.ndcgs else 0.0,
            "precision": float(np.mean(self.precisions)) if self.precisions else 0.0,
            "recall": float(np.mean(self.recalls)) if self.recalls else 0.0,
            "f1_score": float(np.mean(self.f1_scores)) if self.f1_scores else 0.0,
        }

class RetrievalEvaluator:
    """A class to evaluate the performance of a retrieval algorithm."""
    def __init__(self, recommender: HybridRecommender):
        self.recommender = recommender

    def evaluate(
        self,
        evaluation_dataset: List[Dict[str, Any]],
        params: RecommendationParams,
        resource_metadata: List[Dict[str, Any]],
        resource_embeddings: Optional[np.ndarray],
        bm25_index: Optional[Any] = None,
    ) -> RetrievalEvaluationMetrics:
        """Evaluates the recommender on the given dataset."""
        metrics = RetrievalEvaluationMetrics()
        
        if resource_embeddings is None:
            print("Warning: resource_embeddings is None. Cannot perform semantic search.")

        for item in evaluation_dataset:
            query = item["query"]
            relevant_docs_set = set(item["relevant_docs"])

            recommendations = self.recommender.recommend(
                query=query,
                params=params,
                resource_metadata=resource_metadata,
                resource_embeddings=resource_embeddings,
                bm25_index=bm25_index,
            )
            
            retrieved_ids = [rec[0].get('entry_id') for rec in recommendations]

            hit_rate, mrr, ndcg = self._calculate_ranking_metrics(retrieved_ids, relevant_docs_set)
            metrics.update_ranking_metrics(hit_rate, mrr, ndcg)
            
            precision, recall, f1 = self._calculate_classification_metrics(retrieved_ids, relevant_docs_set)
            metrics.update_classification_metrics(precision, recall, f1)

        return metrics

    def _calculate_ranking_metrics(
        self, retrieved_ids: List[str], relevant_docs: Set[str]
    ) -> Tuple[float, float, float]:
        """Calculates Hit Rate, MRR, and nDCG."""
        valid_retrieved_ids = [doc_id for doc_id in retrieved_ids if doc_id is not None]
        if not relevant_docs: return 0.0, 0.0, 0.0
        hit = 1.0 if any(doc_id in relevant_docs for doc_id in valid_retrieved_ids) else 0.0
        mrr = 0.0
        for i, doc_id in enumerate(valid_retrieved_ids):
            if doc_id in relevant_docs:
                mrr = 1.0 / (i + 1)
                break
        dcg = sum(1.0 / np.log2(i + 2) for i, doc_id in enumerate(valid_retrieved_ids) if doc_id in relevant_docs)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(valid_retrieved_ids), len(relevant_docs))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return hit, mrr, ndcg
        
    def _calculate_classification_metrics(
        self, retrieved_ids: List[str], relevant_docs: Set[str]
    ) -> Tuple[float, float, float]:
        """Calculates Precision, Recall, and F1-Score."""
        valid_retrieved_ids = {doc_id for doc_id in retrieved_ids if doc_id is not None}
        true_positives = len(valid_retrieved_ids.intersection(relevant_docs))
        false_positives = len(valid_retrieved_ids) - true_positives
        false_negatives = len(relevant_docs) - true_positives
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1_score

    def print_results(self, metrics: RetrievalEvaluationMetrics):
        """Prints the evaluation results in a formatted way."""
        avg_metrics = metrics.get_average_metrics()
        print("--- Retrieval Evaluation Results ---")
        print("\n--- Ranking Metrics ---")
        print(f"  HIT_RATE: {avg_metrics.get('hit_rate', 0.0):.4f}")
        print(f"  MRR:      {avg_metrics.get('mrr', 0.0):.4f}")
        print(f"  NDCG:     {avg_metrics.get('ndcg', 0.0):.4f}")
        print("\n--- Classification Metrics (from Confusion Matrix) ---")
        print(f"  PRECISION: {avg_metrics.get('precision', 0.0):.4f}")
        print(f"  RECALL:    {avg_metrics.get('recall', 0.0):.4f}")
        print(f"  F1-SCORE:  {avg_metrics.get('f1_score', 0.0):.4f}")
        print("\n------------------------------------")