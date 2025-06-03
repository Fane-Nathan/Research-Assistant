# Evaluation Scripts Documentation

This document provides detailed documentation for the evaluation scripts available in the Research Assistant project.

## 📋 Available Scripts

### 1. Bootstrap Evaluation Set
**Script:** `scripts/bootstrap_evaluation_set.py`

Creates evaluation datasets from your document collection by generating question-answer pairs.

```bash
python scripts/bootstrap_evaluation_set.py \
  --num-questions 100 \
  --difficulty-levels easy,medium,hard \
  --output evaluation_sets/bootstrap_qa.json
```

**Parameters:**
- `--num-questions`: Number of questions to generate
- `--difficulty-levels`: Comma-separated difficulty levels
- `--output`: Output file path for the evaluation dataset

### 2. Interactive Evaluation Labeler
**Script:** `scripts/interactive_evaluation_labeler.py`

Provides a user interface for manually evaluating system responses.

```bash
# Launch interactive evaluation interface
python scripts/interactive_evaluation_labeler.py

# Evaluate specific query set
python scripts/interactive_evaluation_labeler.py \
  --query-set evaluation_queries.json \
  --output results/human_eval_results.json
```

**Features:**
- Real-time response evaluation
- Multiple evaluation criteria
- Export results to JSON format

### 3. Score Review Candidates
**Script:** `scripts/score_review_candidates.py`

Evaluates potential improvements by scoring different model configurations.

```bash
python scripts/score_review_candidates.py \
  --input evaluation_sets/bootstrap_qa.json \
  --model-configs configs/model_variants.json \
  --output results/candidate_scores.json
```

**Use Cases:**
- A/B testing different configurations
- Comparing model performance
- Identifying best-performing setups

## 🔧 Configuration

All scripts support configuration through:
- Command-line arguments
- Configuration files (JSON/YAML)
- Environment variables

## 📊 Output Formats

Evaluation results are typically saved as:
- **JSON**: Structured data for programmatic analysis
- **CSV**: Tabular data for spreadsheet analysis
- **Logs**: Detailed execution information

## 🛠️ Custom Scripts

You can create custom evaluation scripts by extending the base evaluator classes:

```python
from hybrid_search_rag.evaluation.evaluator import Evaluator

class CustomEvaluator(Evaluator):
    def custom_metric(self, predictions, ground_truth):
        # Your custom evaluation logic
        pass
```

## 📚 Related Documentation

- [Evaluation Framework](../testing/evaluation-framework.md)
- [Metrics Definitions](./metrics.md)
- [Performance Optimization](../performance/optimization.md)
