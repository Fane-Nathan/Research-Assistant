"""
Evaluation module for assessing RAG system performance.

This module provides metrics and tools for evaluating both retrieval quality
and generation quality of the RAG system.
"""

from .generation_metrics import GenerationEvaluationMetrics

__all__ = [
    "GenerationEvaluationMetrics",
]