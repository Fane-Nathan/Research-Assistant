# Evaluation Framework

Research Assistant includes a comprehensive evaluation framework to measure and improve the quality of research responses. This guide covers the evaluation methods, metrics, and tools available for assessing performance.

## 🎯 Overview

The evaluation framework provides several approaches to measure:
- **Retrieval Quality** - How well relevant documents are found
- **Response Accuracy** - Correctness of generated answers
- **Citation Reliability** - Accuracy of source attributions
- **User Satisfaction** - Overall usefulness of responses

## 📊 Evaluation Methods

### 1. **Automated Evaluation**

#### Retrieval Metrics
```python
from hybrid_search_rag.evaluation.evaluator import Evaluator
from hybrid_search_rag.evaluation.metrics import RetrievalMetrics

evaluator = Evaluator()
metrics = RetrievalMetrics()

# Evaluate retrieval performance
results = evaluator.evaluate_retrieval(
    queries=test_queries,
    ground_truth=gold_standard_docs,
    top_k=5
)

print(f"Precision@5: {results.precision_at_k}")
print(f"Recall@5: {results.recall_at_k}")
print(f"NDCG@5: {results.ndcg_at_k}")
```

#### Response Quality Metrics
```python
from hybrid_search_rag.evaluation.metrics import ResponseMetrics

response_metrics = ResponseMetrics()

# Evaluate generated responses
scores = response_metrics.evaluate_responses(
    questions=test_questions,
    generated_answers=model_responses,
    reference_answers=gold_standard_answers
)

print(f"BLEU Score: {scores.bleu}")
print(f"ROUGE-L: {scores.rouge_l}")
print(f"Semantic Similarity: {scores.semantic_similarity}")
```

### 2. **Human Evaluation**

#### Interactive Evaluation Tool
```bash
# Launch interactive evaluation interface
python scripts/interactive_evaluation_labeler.py

# Evaluate specific query set
python scripts/interactive_evaluation_labeler.py \
  --query-set evaluation_queries.json \
  --output results/human_eval_results.json
```

#### Evaluation Categories
- **Relevance** (1-5): How well the response answers the question
- **Accuracy** (1-5): Factual correctness of the information
- **Completeness** (1-5): Comprehensiveness of the answer
- **Clarity** (1-5): How well-written and understandable the response is
- **Source Quality** (1-5): Reliability and appropriateness of cited sources

### 3. **Benchmark Evaluation**

#### Standard Datasets
```python
from hybrid_search_rag.evaluation.benchmarks import BenchmarkEvaluator

# Evaluate on academic QA datasets
benchmark = BenchmarkEvaluator()

# SciQ dataset evaluation
sciq_results = benchmark.evaluate_on_sciq(
    model=your_model,
    subset="test"
)

# Custom academic dataset
custom_results = benchmark.evaluate_on_dataset(
    dataset_path="path/to/your/dataset.json",
    model=your_model
)
```

## 🛠️ Evaluation Tools

### 1. **Bootstrap Evaluation Set**
Create evaluation datasets from your document collection:

```bash
python scripts/bootstrap_evaluation_set.py \
  --num-questions 100 \
  --difficulty-levels easy,medium,hard \
  --output evaluation_sets/bootstrap_qa.json
```

### 2. **Score Review Candidates**
Evaluate potential improvements:

```bash
python scripts/score_review_candidates.py \
  --input evaluation_sets/bootstrap_qa.json \
  --model-configs configs/model_variants.json \
  --output results/candidate_scores.json
```

### 3. **Continuous Evaluation**
Set up ongoing evaluation monitoring:

```python
from hybrid_search_rag.evaluation.continuous_eval import ContinuousEvaluator

# Set up continuous monitoring
evaluator = ContinuousEvaluator(
    eval_interval="daily",
    metrics=["precision", "recall", "user_satisfaction"],
    alert_thresholds={"precision": 0.7, "recall": 0.6}
)

evaluator.start_monitoring()
```

## 📈 Performance Metrics

### Retrieval Metrics

| Metric | Description | Formula | Good Score |
|--------|-------------|---------|------------|
| **Precision@K** | Fraction of retrieved docs that are relevant | TP / (TP + FP) | > 0.7 |
| **Recall@K** | Fraction of relevant docs that are retrieved | TP / (TP + FN) | > 0.6 |
| **NDCG@K** | Normalized discounted cumulative gain | DCG / IDCG | > 0.8 |
| **MRR** | Mean reciprocal rank | 1/rank of first relevant doc | > 0.5 |

### Response Quality Metrics

| Metric | Description | Range | Good Score |
|--------|-------------|--------|------------|
| **BLEU** | N-gram overlap with reference | 0-1 | > 0.3 |
| **ROUGE-L** | Longest common subsequence | 0-1 | > 0.4 |
| **BERTScore** | Semantic similarity using BERT | 0-1 | > 0.8 |
| **Faithfulness** | Consistency with source docs | 0-1 | > 0.9 |

## 🔧 Custom Evaluation Setup

### 1. **Define Evaluation Dataset**
```json
{
  "questions": [
    {
      "id": "q001",
      "question": "What is the attention mechanism in transformers?",
      "category": "architecture",
      "difficulty": "medium",
      "reference_answer": "The attention mechanism...",
      "relevant_docs": ["doc1", "doc2", "doc3"]
    }
  ]
}
```

### 2. **Configure Evaluation Pipeline**
```python
from hybrid_search_rag.evaluation.pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(
    retrieval_metrics=["precision@5", "recall@5", "ndcg@5"],
    generation_metrics=["bleu", "rouge", "bertscore"],
    human_eval_enabled=True,
    benchmark_datasets=["sciq", "msmarco"]
)

results = pipeline.run_evaluation(
    test_queries="evaluation_sets/test_queries.json",
    output_dir="results/evaluation_run_2024"
)
```

### 3. **Analysis and Reporting**
```python
from hybrid_search_rag.evaluation.analysis import EvaluationAnalyzer

analyzer = EvaluationAnalyzer()

# Generate comprehensive report
report = analyzer.generate_report(
    results_dir="results/evaluation_run_2024",
    include_plots=True,
    output_format="html"
)

# Compare different model configurations
comparison = analyzer.compare_models([
    "results/baseline_model",
    "results/improved_model",
    "results/fine_tuned_model"
])
```

## 🎯 Improvement Strategies

### Based on Evaluation Results

#### Low Retrieval Scores
- **Increase document diversity** in knowledge base
- **Improve embedding quality** (try different models)
- **Tune retrieval parameters** (similarity thresholds, top-k values)
- **Enhance query processing** (query expansion, normalization)

#### Low Response Quality
- **Improve prompting strategy** (better system prompts)
- **Try different LLM providers** or models
- **Increase context window** for more relevant information
- **Fine-tune response generation** parameters

#### Poor Source Citations
- **Improve document chunking** strategy
- **Add citation training** examples
- **Implement citation validation** checks
- **Enhance source tracking** throughout pipeline

## 🔄 Continuous Improvement Cycle

1. **Collect Performance Data** - Regular evaluation runs
2. **Identify Issues** - Analysis of failure cases
3. **Implement Improvements** - Code/configuration changes
4. **A/B Testing** - Compare improvements against baseline
5. **Deploy Best Performers** - Update production system
6. **Monitor Results** - Ongoing performance tracking

## 📚 Further Reading

- [Performance Optimization](../performance/optimization.md)
- [General Troubleshooting](../reference/troubleshooting.md)