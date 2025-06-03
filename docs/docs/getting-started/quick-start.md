# Quick Start Guide

Get Research Assistant up and running in just a few minutes! This guide assumes you've already completed the [installation](./installation.md).

## 🌐 Try the Live Demo First

**Want to test Research Assistant immediately?** Try our live web interface:
[https://research-assistant-flx.streamlit.app/](https://research-assistant-flx.streamlit.app/)

No installation required - experience the full functionality including search, analysis, and AI-powered insights!

---

## 🚀 Launch Your First Research Session (Local Setup)

### Step 1: Activate Your Environment

```bash
# If using conda
conda activate research_env

# If using venv
# Windows
research_env\Scripts\activate
# macOS/Linux
source research_env/bin/activate
```

### Step 2: Basic Configuration Check

Verify your setup is working:

```bash
cd research-assistant
python -c "from hybrid_search_rag.config import run_config_checks; run_config_checks()"
```

### Step 3: Choose Your Interface

Research Assistant offers multiple ways to interact with the system:

#### Option A: Command Line Interface (Fastest)

```bash
# Navigate to scripts directory
cd scripts

# Get help
python cli.py --help

# Run a simple query
python cli.py query "What are the latest developments in transformer models?"
```

#### Option B: Web Interface (Most User-Friendly)

```bash
# Launch the Streamlit web app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

#### Option C: Python API (Most Flexible)

```python
from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import HybridRecommender
from hybrid_search_rag.llm_services.llm_interface import LLMInterface

# Initialize components
retriever = HybridRecommender()
llm = LLMInterface()

# Perform a query
query = "Explain attention mechanisms in neural networks"
results = retriever.search(query, top_k=5)
answer = llm.generate_response(query, results)
print(answer)
```

## 📊 Your First Data Collection

If you're starting fresh, you'll want to collect some academic papers:

### Quick arXiv Collection

```bash
# Fetch recent papers from Computer Science
python scripts/cli.py fetch-arxiv --query "cat:cs.AI" --max-results 50

# Fetch papers on a specific topic
python scripts/cli.py fetch-arxiv --query "attention mechanism neural networks" --max-results 20
```

### Check Your Data

```bash
# View collected papers
python scripts/cli.py list-papers

# Show statistics
python scripts/cli.py stats
```

## 🔍 Your First Query

Now that you have some data, let's ask a question:

### Using the CLI

```bash
python scripts/cli.py query "What are the main components of a transformer architecture?"
```

### Using the Web Interface

1. Open `http://localhost:8501` in your browser
2. Type your question in the query box
3. Click "Search" and wait for results
4. Explore the retrieved papers and generated answer

### Expected Output

You should see:
- **Generated Answer**: AI-generated response based on your papers
- **Source Papers**: Relevant papers that were used to generate the answer
- **Confidence Scores**: How relevant each source is to your query

## 🛠️ Common First-Time Tasks

### Add More Data Sources

```bash
# Fetch from specific arXiv categories
python scripts/cli.py fetch-arxiv --query "cat:cs.CL" --max-results 100  # Computational Linguistics
python scripts/cli.py fetch-arxiv --query "cat:cs.LG" --max-results 100  # Machine Learning

# Fetch papers by specific authors
python scripts/cli.py fetch-arxiv --query "au:Vaswani" --max-results 50

# Fetch from web URLs (if you have specific papers)
python scripts/cli.py fetch-url "https://arxiv.org/abs/1706.03762"  # Attention Is All You Need
```

### Explore Different Query Types

```bash
# Technical explanations
python scripts/cli.py query "How does BERT differ from GPT?"

# Literature reviews
python scripts/cli.py query "What are the recent advances in few-shot learning?"

# Methodology questions
python scripts/cli.py query "What evaluation metrics are used for text summarization?"

# Comparative analysis
python scripts/cli.py query "Compare different approaches to neural machine translation"
```

### Customize Your Search

```bash
# Adjust the number of retrieved papers
python scripts/cli.py query "transformer architecture" --top-k 10

# Use different retrieval modes
python scripts/cli.py query "attention mechanism" --mode semantic  # Semantic search only
python scripts/cli.py query "attention mechanism" --mode keyword   # Keyword search only
python scripts/cli.py query "attention mechanism" --mode hybrid    # Both (default)
```

## 📈 Monitor Your System

### Check System Status

```bash
# View system statistics
python scripts/cli.py stats

# Check data integrity
python scripts/cli.py verify-data

# View recent activity
tail -f logs/app.log
```

### Performance Tips

1. **Start Small**: Begin with 50-100 papers to test the system
2. **Use Specific Queries**: More specific questions yield better results
3. **Monitor Memory**: Large datasets require more RAM
4. **Cache Results**: The system automatically caches embeddings and indexes

## 🎯 What's Next?

Now that you have Research Assistant running:

### Immediate Next Steps
1. **[Configure Advanced Settings](./configuration.md)** - Set up API keys, adjust parameters
2. **[Learn the CLI](../interfaces/cli.md)** - Master all command-line features
3. **[Explore the Web Interface](../interfaces/streamlit-app.md)** - Use the interactive interface

### Dive Deeper
1. **[Understand the Architecture](../architecture/overview.md)** - Learn how it all works
2. **[Optimize Performance](../performance/optimization.md)** - Scale for larger datasets
3. **[Contribute](../development/contributing.md)** - Help improve the project

### Advanced Use Cases
1. **[Custom Data Sources](../data/data-sources.md)** - Add your own papers and documents
2. **[API Integration](../api/external-apis.md)** - Integrate with other tools
3. **[Evaluation & Testing](../testing/evaluation-framework.md)** - Measure and improve performance

## 🆘 Need Help?

### Quick Debugging

```bash
# Check configuration
python -c "from hybrid_search_rag.config import *; print('Config OK')"

# Test API connections
python scripts/cli.py test-apis

# Verify data integrity
python scripts/cli.py verify-data
```

### Common Issues

| Issue | Solution |
|-------|----------|
| No papers found | Run `python scripts/cli.py fetch-arxiv --query "cat:cs.AI" --max-results 50` |
| API errors | Check your `.env` file for correct API keys |
| Memory errors | Reduce batch sizes or use a machine with more RAM |
| Slow queries | Start with smaller datasets and optimize gradually |

### Getting Support

- 📖 **Documentation**: Browse the full documentation
- 🐛 **Bug Reports**: Create an issue on GitHub
- 💬 **Discussions**: Join the community forums
- 📧 **Contact**: Reach out to the development team

---

**Congratulations!** 🎉 You've successfully run your first Research Assistant session. You're now ready to explore the full power of AI-enhanced research!

*Happy researching!* 🔬✨
