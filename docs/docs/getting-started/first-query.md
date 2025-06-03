# Your First Query

Now that you have Research Assistant installed and configured, let's run your first query and explore the results!

## 🎯 Understanding Query Types

Research Assistant can handle various types of research queries:

### 📚 **Conceptual Questions**
- "What is a transformer neural network?"
- "Explain the attention mechanism in deep learning"
- "How does BERT differ from GPT?"

### 🔬 **Technical Deep Dives**
- "What are the mathematical foundations of self-attention?"
- "How is backpropagation implemented in transformer models?"
- "What optimization techniques are used in large language model training?"

### 📊 **Literature Reviews**
- "What are recent advances in few-shot learning?"
- "Compare different approaches to neural machine translation"
- "What evaluation metrics are used for text summarization?"

### 🎯 **Specific Paper Searches**
- "Find papers about BERT fine-tuning techniques"
- "Show me research on attention visualization methods"
- "What papers discuss GPT-4 capabilities?"

## 🚀 Running Your First Query

### Method 1: Command Line Interface

Open your terminal and navigate to the Research Assistant directory:

```bash
cd research-assistant
conda activate research_env  # or your environment name
```

#### Basic Query
```bash
python scripts/cli.py query "What is a transformer neural network?"
```

#### Query with Options
```bash
python scripts/cli.py query "attention mechanism in neural networks" \
  --top-k 5 \
  --mode hybrid \
  --format detailed
```

### Method 2: Web Interface

1. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Enter your query** in the text box

4. **Click "Search"** and wait for results

### Method 3: Python API

```python
from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import HybridRecommender
from hybrid_search_rag.llm_services.llm_interface import LLMInterface
from hybrid_search_rag.data_handling.data_manager import DataManager

# Initialize components
data_manager = DataManager()
retriever = HybridRecommender(data_manager)
llm = LLMInterface()

# Your query
query = "What are the key innovations in transformer architecture?"

# Retrieve relevant documents
retrieved_docs = retriever.search(query, top_k=5)

# Generate response
response = llm.generate_response(query, retrieved_docs)

print("Query:", query)
print("\\nGenerated Answer:")
print(response.answer)
print("\\nSources:")
for i, doc in enumerate(response.sources, 1):
    print(f"{i}. {doc.title} - {doc.authors}")
```

## 📋 Understanding the Results

When you run a query, Research Assistant returns several types of information:

### 1. **Generated Answer**
An AI-generated response based on the retrieved academic papers:

```
The transformer architecture introduced several key innovations that revolutionized 
natural language processing:

1. **Self-Attention Mechanism**: Unlike RNNs, transformers use self-attention to 
   process all positions in a sequence simultaneously, enabling parallelization 
   and better capture of long-range dependencies.

2. **Multi-Head Attention**: The architecture employs multiple attention heads 
   that can focus on different types of relationships within the input sequence.

3. **Positional Encoding**: Since transformers don't have inherent sequence order, 
   positional encodings are added to input embeddings to provide position information.
   
[... additional content based on retrieved papers ...]
```

### 2. **Source Documents**
Papers that were used to generate the answer:

```
📄 Sources:
1. "Attention Is All You Need" - Vaswani et al. (2017)
   Relevance: 0.92 | arXiv:1706.03762
   
2. "BERT: Pre-training of Deep Bidirectional Transformers" - Devlin et al. (2018)
   Relevance: 0.87 | arXiv:1810.04805
   
3. "The Annotated Transformer" - Rush (2018)
   Relevance: 0.84 | http://nlp.seas.harvard.edu/annotated-transformer/
```

### 3. **Retrieval Information**
Details about how the search was performed:

```
🔍 Search Details:
- Query Type: Hybrid (Semantic + Keyword)
- Documents Searched: 150
- Top-K Retrieved: 5
- Processing Time: 2.3 seconds
- Embedding Time: 0.8 seconds
- Generation Time: 1.5 seconds
```

## 🎛️ Customizing Your Query

### Retrieval Mode Options

```bash
# Semantic search only (uses embeddings)
python scripts/cli.py query "transformer architecture" --mode semantic

# Keyword search only (uses BM25)
python scripts/cli.py query "transformer architecture" --mode keyword

# Hybrid search (combines both - recommended)
python scripts/cli.py query "transformer architecture" --mode hybrid
```

### Number of Retrieved Documents

```bash
# Retrieve more documents for comprehensive answers
python scripts/cli.py query "attention mechanism" --top-k 10

# Quick answers with fewer sources
python scripts/cli.py query "attention mechanism" --top-k 3
```

### Output Format Options

```bash
# Detailed format (default)
python scripts/cli.py query "BERT architecture" --format detailed

# Concise format
python scripts/cli.py query "BERT architecture" --format concise

# JSON format (for programmatic use)
python scripts/cli.py query "BERT architecture" --format json
```

### Advanced Options

```bash
# Include confidence scores
python scripts/cli.py query "GPT training" --show-scores

# Filter by publication year
python scripts/cli.py query "language models" --year-filter 2020-2024

# Filter by specific arXiv categories
python scripts/cli.py query "neural networks" --categories cs.AI,cs.LG
```

## 🎯 Example Query Session

Let's walk through a complete example:

### Step 1: Ask a Conceptual Question

```bash
python scripts/cli.py query "How does the attention mechanism work in transformers?"
```

**Expected Result:**
- A detailed explanation of attention mechanisms
- References to key papers (Vaswani et al., Bahdanau et al.)
- Mathematical foundations and intuitive explanations

### Step 2: Follow-up with a Technical Question

```bash
python scripts/cli.py query "What is the mathematical formula for self-attention?" --top-k 3
```

**Expected Result:**
- Mathematical equations and formulas
- Step-by-step derivations
- Implementation details

### Step 3: Explore Recent Developments

```bash
python scripts/cli.py query "What are recent improvements to transformer architectures?" --year-filter 2022-2024
```

**Expected Result:**
- Recent innovations and papers
- Comparative analysis of improvements
- Performance benchmarks

## 🔍 Analyzing Query Performance

### Understanding Relevance Scores

Relevance scores (0.0 to 1.0) indicate how well each document matches your query:
- **0.9-1.0**: Highly relevant, directly answers your question
- **0.7-0.8**: Very relevant, contains important related information
- **0.5-0.6**: Moderately relevant, may contain useful context
- **Below 0.5**: Less relevant, but may provide background information

### Query Processing Time

Monitor performance components:
- **Embedding Time**: Time to convert your query to a vector
- **Search Time**: Time to find relevant documents
- **Generation Time**: Time for the LLM to create the response
- **Total Time**: End-to-end query processing

### Improving Query Results

#### If Results Are Too General:
- Make your query more specific
- Use technical terminology
- Include specific paper titles or authors
- Increase `--top-k` to get more context

#### If Results Are Too Narrow:
- Use broader terminology
- Include synonyms or related concepts
- Use semantic search mode
- Decrease `--top-k` for focused answers

#### If Results Are Outdated:
- Use `--year-filter` to focus on recent papers
- Fetch newer papers from arXiv
- Update your dataset regularly

## 🎉 Success! You've Run Your First Query

Congratulations! You've successfully:
- ✅ Executed your first research query
- ✅ Understood the result structure
- ✅ Learned about customization options
- ✅ Explored different query types

## 🚀 What's Next?

Now that you've mastered basic querying:

### Immediate Next Steps
1. **[Master the CLI](../interfaces/cli.md)** - Learn all available commands
2. **[Explore the Web Interface](../interfaces/streamlit-app.md)** - Try the visual interface
3. **[Add More Data](../data/data-sources.md)** - Expand your paper collection

### Advanced Usage
1. **[Understand Hybrid Search](../architecture/hybrid-search.md)** - Learn how retrieval works
2. **[Optimize Performance](../performance/optimization.md)** - Scale for larger datasets
3. **[Evaluate Results](../testing/evaluation-framework.md)** - Measure and improve quality

### Integration & Development
1. **[API Integration](../api/external-apis.md)** - Integrate with other tools
2. **[Custom Data Sources](../data/data-sources.md)** - Add your own documents
3. **[Contribute](../development/contributing.md)** - Help improve the project

## 📞 Need Help?

### Quick Debugging

```bash
# Check if your data is loaded
python scripts/cli.py stats

# Test API connections
python scripts/cli.py test-apis

# View recent queries and results
tail -f logs/app.log
```

### Common Issues

| Problem | Solution |
|---------|----------|
| "No documents found" | Run `python scripts/cli.py fetch-arxiv --query "cat:cs.AI" --max-results 50` |
| API key errors | Check your `.env` file configuration |
| Slow responses | Start with smaller datasets or adjust `--top-k` |
| Generic answers | Make queries more specific or add more relevant data |

### Getting Support

- 📖 **Documentation**: Continue reading the guides
- 🔧 **Troubleshooting**: Check the [troubleshooting guide](../reference/troubleshooting.md)
- 💬 **Community**: Join discussions and ask questions
- 🐛 **Bug Reports**: Report issues on GitHub

---

**Congratulations!** 🎊 You've successfully completed your first Research Assistant query session. You're now ready to dive deeper into AI-powered research!

*Happy exploring!* 🔬🚀
