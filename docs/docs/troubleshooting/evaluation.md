# Troubleshooting Evaluation Issues

This guide helps you diagnose and fix common problems when running evaluation scripts and frameworks.

## 🚨 Common Issues

### 1. Import Errors

#### "Module not found" errors
```
ModuleNotFoundError: No module named 'hybrid_search_rag'
```

**Solution:**
```bash
# Ensure you're in the project root directory
cd /path/to/Research-Assistant

# Install dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### Missing evaluation dependencies
```
ImportError: No module named 'nltk' or 'scikit-learn'
```

**Solution:**
```bash
# Install evaluation-specific dependencies
pip install nltk scikit-learn numpy pandas

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Data Issues

#### Empty evaluation dataset
```
ValueError: Cannot evaluate with empty dataset
```

**Diagnosis:**
- Check if evaluation files exist
- Verify file format (JSON/CSV)
- Ensure proper data structure

**Solution:**
```bash
# Generate evaluation dataset
python scripts/bootstrap_evaluation_set.py --num-questions 50

# Verify dataset structure
python -c "import json; print(json.load(open('evaluation_sets/bootstrap_qa.json'))[:2])"
```

#### Inconsistent data format
```
KeyError: 'question' or 'ground_truth'
```

**Fix:**
Ensure your evaluation data follows this structure:
```json
{
  "questions": [
    {
      "id": "q001",
      "question": "What is...",
      "ground_truth": "The answer is...",
      "category": "technical",
      "difficulty": "medium"
    }
  ]
}
```

### 3. Performance Issues

#### Slow evaluation runs
**Symptoms:**
- Scripts taking hours to complete
- High memory usage
- System becoming unresponsive

**Solutions:**
```bash
# Reduce batch size
python scripts/bootstrap_evaluation_set.py --batch-size 10

# Use subset for testing
python scripts/score_review_candidates.py --max-questions 20

# Enable caching
export ENABLE_EVALUATION_CACHE=true
```

#### Memory errors
```
MemoryError: Unable to allocate array
```

**Solutions:**
- Reduce embedding dimensions
- Process data in smaller batches
- Use sparse matrices where possible
- Clear cache between runs

### 4. Model/API Issues

#### API rate limiting
```
RateLimitError: Too many requests
```

**Solutions:**
```python
# Add delays between requests
import time
time.sleep(1)  # Add in your evaluation loop

# Use batch processing
# Configure in evaluation config
{
  "api_delay": 1.0,
  "batch_size": 5,
  "max_retries": 3
}
```

#### Model loading failures
```
OSError: Unable to load model
```

**Diagnosis:**
```bash
# Check model files
ls -la models/
du -sh models/*

# Test model loading
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 5. Metric Calculation Errors

#### Division by zero in metrics
```
ZeroDivisionError: division by zero
```

**Common causes:**
- No relevant documents found
- Empty ground truth
- Malformed evaluation data

**Prevention:**
```python
# Add checks in custom metrics
def safe_precision(relevant, retrieved):
    if len(retrieved) == 0:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(retrieved)
```

#### NaN values in results
**Debug:**
```python
import pandas as pd
results_df = pd.read_csv('evaluation_results.csv')
print(results_df.isnull().sum())  # Check for NaN values
print(results_df.describe())      # Statistical summary
```

## 🔧 Debugging Tools

### 1. Verbose Logging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/bootstrap_evaluation_set.py --verbose
```

### 2. Step-by-step Execution
```python
# Test individual components
from hybrid_search_rag.evaluation.evaluator import Evaluator

evaluator = Evaluator()
# Test retrieval
results = evaluator.test_retrieval("sample query")
print(f"Retrieved: {len(results)} documents")

# Test metrics
precision = evaluator.calculate_precision(retrieved, relevant)
print(f"Precision: {precision}")
```

### 3. Data Validation Scripts
```bash
# Validate evaluation dataset
python -c "
import json
data = json.load(open('evaluation_sets/bootstrap_qa.json'))
print(f'Questions: {len(data.get(\"questions\", []))}')
for q in data['questions'][:3]:
    print(f'ID: {q.get(\"id\")}, Question: {q.get(\"question\")[:50]}...')
"
```

## 📊 Performance Monitoring

### System Resource Monitoring
```bash
# Monitor during evaluation
# Terminal 1: Run evaluation
python scripts/bootstrap_evaluation_set.py

# Terminal 2: Monitor resources
watch -n 1 "ps aux | grep python | head -5; free -h; df -h"
```

### Evaluation Progress Tracking
```python
# Add progress bars
from tqdm import tqdm

for question in tqdm(questions, desc="Evaluating"):
    # Your evaluation code
    result = evaluate_question(question)
```

## 🛠️ Configuration Troubleshooting

### Environment Variables
```bash
# Check current environment
env | grep -E "(API_KEY|MODEL|EVAL)"

# Set required variables
export GOOGLE_API_KEY="your_key"
export GROQ_API_KEY="your_key"
export EVALUATION_DATA_DIR="./evaluation_sets"
```

### Config File Issues
```yaml
# evaluation_config.yaml - correct format
retrieval:
  top_k: 5
  method: "hybrid"
  
generation:
  model: "google/gemini-pro"
  max_tokens: 1000
  
metrics:
  - "precision@5"
  - "recall@10"
  - "ndcg@5"
```

## 🔄 Recovery Procedures

### Corrupted Evaluation Data
```bash
# Backup existing data
cp evaluation_sets/bootstrap_qa.json evaluation_sets/bootstrap_qa.json.backup

# Regenerate from scratch
rm evaluation_sets/bootstrap_qa.json
python scripts/bootstrap_evaluation_set.py --num-questions 100
```

### Failed Evaluation Runs
```python
# Resume from checkpoint
evaluator = Evaluator(checkpoint_dir="./checkpoints")
evaluator.resume_evaluation("run_20241201_143022")
```

## 📞 Getting Additional Help

### Log Analysis
When reporting issues, include:
- Full error traceback
- System information (`python --version`, OS)
- Configuration files
- Sample data that causes the issue

### Debug Information Script
```bash
# Generate debug info
python -c "
import sys, platform, pkg_resources
print('Python:', sys.version)
print('Platform:', platform.platform())
print('Packages:')
for pkg in ['nltk', 'numpy', 'pandas', 'scikit-learn']:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  {pkg}: {version}')
    except:
        print(f'  {pkg}: NOT INSTALLED')
"
```

## 📚 Related Documentation

- [Evaluation Framework](../testing/evaluation-framework.md)
- [Evaluation Scripts](../reference/evaluation-scripts.md)
- [Metrics Definitions](../reference/metrics.md)
- [General Troubleshooting](../reference/troubleshooting.md)
