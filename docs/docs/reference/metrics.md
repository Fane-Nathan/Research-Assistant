# Metrics Definitions

This document provides detailed definitions of all evaluation metrics used in the Research Assistant evaluation framework.

## 🎯 Retrieval Metrics

### Precision@K
Measures the proportion of retrieved documents that are relevant.

**Formula:** `Precision@K = (Relevant Retrieved Documents) / K`

**Range:** 0.0 to 1.0 (higher is better)

**Use Case:** When you want to ensure that most retrieved documents are useful.

### Recall@K  
Measures the proportion of relevant documents that are retrieved.

**Formula:** `Recall@K = (Relevant Retrieved Documents) / (Total Relevant Documents)`

**Range:** 0.0 to 1.0 (higher is better)

**Use Case:** When you want to ensure you don't miss important documents.

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality, considering both relevance and position.

**Formula:** Complex ranking-based calculation that rewards relevant documents appearing higher in results.

**Range:** 0.0 to 1.0 (higher is better)

**Use Case:** When document ranking order matters significantly.

### Mean Reciprocal Rank (MRR)
Measures the average of reciprocal ranks of the first relevant document.

**Formula:** `MRR = (1/N) * Σ(1/rank_i)`

**Range:** 0.0 to 1.0 (higher is better)

**Use Case:** When finding the first relevant result quickly is important.

## 📝 Generation Metrics

### BLEU Score
Measures n-gram overlap between generated and reference text.

**Range:** 0.0 to 1.0 (higher is better)

**Strengths:** Good for fluency assessment
**Limitations:** May not capture semantic similarity well

### ROUGE-L
Measures longest common subsequence between generated and reference text.

**Range:** 0.0 to 1.0 (higher is better)

**Use Case:** Summarization and content generation evaluation

### BERTScore
Uses BERT embeddings to measure semantic similarity.

**Range:** Typically 0.0 to 1.0 (higher is better)

**Advantages:** Captures semantic meaning beyond surface-level similarity

### Semantic Similarity
Cosine similarity between sentence embeddings.

**Range:** -1.0 to 1.0 (higher is better)

**Use Case:** Measuring conceptual similarity regardless of exact wording

## 👥 Human Evaluation Metrics

### Relevance (1-5 Scale)
How well the response answers the question.

- **5:** Perfectly answers the question
- **4:** Mostly answers with minor gaps
- **3:** Partially answers, some important aspects missing
- **2:** Tangentially related but inadequate
- **1:** Completely irrelevant

### Accuracy (1-5 Scale)
Factual correctness of the information.

- **5:** All facts are correct
- **4:** Mostly correct with minor inaccuracies
- **3:** Generally correct but some notable errors
- **2:** Several factual errors
- **1:** Mostly or entirely incorrect

### Completeness (1-5 Scale)
Comprehensiveness of the answer.

- **5:** Comprehensive, covers all important aspects
- **4:** Covers most important aspects
- **3:** Covers some aspects but missing key points
- **2:** Incomplete, many important aspects missing
- **1:** Very incomplete or superficial

### Clarity (1-5 Scale)
How well-written and understandable the response is.

- **5:** Exceptionally clear and well-structured
- **4:** Clear and easy to understand
- **3:** Generally clear with some confusing parts
- **2:** Somewhat unclear or poorly organized
- **1:** Very unclear or confusing

### Source Quality (1-5 Scale)
Reliability and appropriateness of cited sources.

- **5:** Excellent, authoritative sources
- **4:** Good, reliable sources
- **3:** Adequate sources
- **2:** Some questionable sources
- **1:** Poor or unreliable sources

## 📊 Composite Metrics

### F1 Score
Harmonic mean of precision and recall.

**Formula:** `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

**Use Case:** When you need a balanced measure of precision and recall.

### Overall Quality Score
Weighted average of multiple human evaluation metrics.

**Formula:** `Quality = w1*Relevance + w2*Accuracy + w3*Completeness + w4*Clarity`

**Customizable:** Weights can be adjusted based on use case priorities.

## 🔧 Metric Selection Guidelines

### For Information Retrieval Systems:
- Primary: Precision@5, Recall@10, NDCG@5
- Secondary: MRR, F1 Score

### For Question Answering Systems:
- Automated: BLEU, ROUGE-L, BERTScore
- Human: Relevance, Accuracy, Completeness

### For Research Applications:
- Emphasis on: Accuracy, Source Quality, Completeness
- Less critical: Fluency metrics (unless user-facing)

## 📈 Benchmarking Targets

### Good Performance Thresholds:
- **Precision@5**: > 0.7
- **Recall@10**: > 0.6
- **NDCG@5**: > 0.65
- **BERTScore**: > 0.85
- **Human Relevance**: > 4.0/5.0
- **Human Accuracy**: > 4.0/5.0

### Excellent Performance Thresholds:
- **Precision@5**: > 0.85
- **Recall@10**: > 0.75
- **NDCG@5**: > 0.8
- **BERTScore**: > 0.9
- **Human Relevance**: > 4.5/5.0
- **Human Accuracy**: > 4.5/5.0

## 📚 Related Documentation

- [Evaluation Framework](../testing/evaluation-framework.md)
- [Evaluation Scripts](./evaluation-scripts.md)
- [Performance Optimization](../performance/optimization.md)
