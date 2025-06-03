# Configuration Guide

This guide covers all configuration options for Research Assistant, from basic API setup to advanced performance tuning.

## 🔑 Essential Configuration

### Environment Variables

Create a `.env` file in your project root with the following variables:

```env
# =====================================
# CORE API KEYS (Required)
# =====================================

# Google Gemini API (Primary LLM and Embeddings)
GOOGLE_API_KEY=your_google_api_key_here

# =====================================
# ALTERNATIVE LLM PROVIDERS (Optional)
# =====================================

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Groq API (Fast inference)
GROQ_API_KEY=your_groq_api_key_here

# Anthropic Claude API
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# =====================================
# EMBEDDING PROVIDERS (Optional)
# =====================================

# Cohere API (Alternative embeddings)
COHERE_API_KEY=your_cohere_api_key_here

# HuggingFace API (Alternative embeddings)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# =====================================
# CUSTOM ENDPOINTS (Optional)
# =====================================

# Custom LLM endpoint (e.g., local deployment)
CUSTOM_LLM_ENDPOINT=https://your-custom-endpoint.com
CUSTOM_LLM_API_KEY=your_custom_api_key

# Custom embedding endpoint
CUSTOM_EMBEDDING_ENDPOINT=https://your-embedding-endpoint.com
CUSTOM_EMBEDDING_API_KEY=your_embedding_api_key

# =====================================
# DATA SOURCES (Optional)
# =====================================

# Semantic Scholar API
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

# IEEE Xplore API
IEEE_API_KEY=your_ieee_api_key

# =====================================
# ADVANCED SETTINGS (Optional)
# =====================================

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Data directory override
DATA_DIR_OVERRIDE=/custom/path/to/data

# Max concurrent requests
MAX_CONCURRENT_REQUESTS=10

# Request timeout (seconds)
REQUEST_TIMEOUT=30
```

### Getting API Keys

#### Google Gemini API
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file

#### OpenAI API
1. Visit [OpenAI API](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key to your `.env` file

#### Groq API
1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up for an account
3. Generate a new API key
4. Copy the key to your `.env` file

## ⚙️ Core Configuration (`config.py`)

The main configuration is managed through `hybrid_search_rag/config.py`. Here are the key settings you can customize:

### Data Storage Settings

```python
# Default data directory
DATA_DIR = "data_hybrid"

# Metadata storage
METADATA_FILE = "combined_metadata.json"
EMBEDDINGS_FILE = "combined_embeddings.npy"
BM25_INDEX_FILE = "bm25_index.pkl"
DOCUMENT_HASHES_FILE = "document_hashes.json"

# Chunk size for text processing
DEFAULT_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

### LLM Provider Configuration

```python
# Primary LLM provider
PRIMARY_LLM_PROVIDER = "google"  # Options: google, openai, groq, anthropic

# Fallback providers (in order)
LLM_FALLBACK_PROVIDERS = ["groq", "openai"]

# Model names
GOOGLE_MODEL_NAME = "gemini-1.5-pro"
OPENAI_MODEL_NAME = "gpt-4o"
GROQ_MODEL_NAME = "llama-3.1-70b-versatile"

# Generation parameters
MAX_TOKENS = 2048
TEMPERATURE = 0.1
TOP_P = 0.9
```

### Embedding Configuration

```python
# Primary embedding provider
PRIMARY_EMBEDDING_PROVIDER = "google"  # Options: google, openai, cohere, huggingface

# Embedding model names
GOOGLE_EMBEDDING_MODEL = "text-embedding-004"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
COHERE_EMBEDDING_MODEL = "embed-english-v3.0"

# Embedding dimensions
EMBEDDING_DIMENSION = 768  # Adjust based on your model
```

### Retrieval Settings

```python
# Hybrid search weights
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Retrieval parameters
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MIN_SIMILARITY_THRESHOLD = 0.1

# BM25 parameters
BM25_K1 = 1.2
BM25_B = 0.75
```

### Rate Limiting

```python
# API rate limits (requests per minute)
GOOGLE_API_RATE_LIMIT = 60
OPENAI_API_RATE_LIMIT = 60
GROQ_API_RATE_LIMIT = 30

# Concurrent request limits
MAX_CONCURRENT_EMBEDDING_REQUESTS = 5
MAX_CONCURRENT_LLM_REQUESTS = 3
MAX_CONCURRENT_FETCH_REQUESTS = 10

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
BACKOFF_FACTOR = 2
```

## 🔧 Advanced Configuration

### Custom Model Endpoints

For using custom or locally deployed models:

```python
# In your .env file
CUSTOM_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
CUSTOM_EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings
```

### Data Source Configuration

Configure arXiv fetching behavior:

```python
# arXiv settings
ARXIV_MAX_RESULTS_PER_QUERY = 100
ARXIV_RETRY_DELAY = 2
ARXIV_CATEGORIES = [
    "cs.AI",  # Artificial Intelligence
    "cs.CL",  # Computational Linguistics
    "cs.LG",  # Machine Learning
    "cs.IR",  # Information Retrieval
]

# Web scraping settings
WEB_SCRAPER_USER_AGENT = "ResearchAssistant/1.0"
WEB_SCRAPER_TIMEOUT = 30
WEB_SCRAPER_RESPECT_ROBOTS = True
```

### Performance Tuning

```python
# Memory management
MAX_MEMORY_USAGE_GB = 8
BATCH_SIZE_EMBEDDINGS = 32
BATCH_SIZE_PROCESSING = 16

# Caching
ENABLE_EMBEDDING_CACHE = True
ENABLE_LLM_RESPONSE_CACHE = True
CACHE_TTL_HOURS = 24

# Indexing
USE_FAISS_INDEX = True  # For large datasets
FAISS_INDEX_TYPE = "IVF"  # Options: Flat, IVF, HNSW
```

## 📁 Directory Structure Configuration

You can customize where Research Assistant stores its data:

```python
# Custom data directories
PROJECT_ROOT = "/path/to/your/project"
DATA_DIR = "/custom/data/location"
LOGS_DIR = "/custom/logs/location"
CACHE_DIR = "/custom/cache/location"
```

## 🔍 Logging Configuration

Customize logging behavior:

```python
# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
LOG_FILE = "logs/research_assistant.log"
MAX_LOG_SIZE_MB = 100
LOG_BACKUP_COUNT = 5

# Component-specific logging
ENABLE_API_LOGGING = True
ENABLE_PERFORMANCE_LOGGING = True
ENABLE_USER_QUERY_LOGGING = False  # Privacy consideration
```

## 🧪 Development Configuration

For development and testing:

```python
# Development settings
DEBUG_MODE = False
ENABLE_PROFILING = False
MOCK_API_RESPONSES = False  # For testing without API calls

# Testing configuration
TEST_DATA_DIR = "tests/data"
TEST_MAX_PAPERS = 10
ENABLE_INTEGRATION_TESTS = True
```

## ✅ Configuration Validation

Run the configuration checker to ensure everything is set up correctly:

```bash
python -c "
from hybrid_search_rag.config import run_config_checks
run_config_checks()
"
```

This will check:
- ✅ Environment variables are loaded
- ✅ API keys are present and valid
- ✅ Data directories exist and are writable
- ✅ Required dependencies are installed
- ✅ Model endpoints are accessible

### Example Output

```
🔍 Running Research Assistant Configuration Checks...

✅ Project Root: /path/to/research-assistant
✅ Environment File: .env loaded successfully
✅ Data Directory: data_hybrid (writable)
✅ Logs Directory: logs (writable)

🔑 API Keys:
✅ Google Gemini API: Valid
✅ OpenAI API: Valid
⚠️  Groq API: Not configured (optional)
⚠️  Anthropic API: Not configured (optional)

🧠 Models:
✅ Primary LLM: google/gemini-1.5-pro (accessible)
✅ Primary Embedding: google/text-embedding-004 (accessible)

📊 Data Status:
✅ Metadata: 150 documents
✅ Embeddings: 150 documents (768 dims)
✅ BM25 Index: Ready
✅ Document Hashes: 150 entries

🚀 System Ready! All core components configured correctly.
```

## 🚨 Troubleshooting Configuration

### Common Issues

#### API Key Issues
```bash
# Test individual API keys
python -c "
from hybrid_search_rag.llm_services.llm_interface import LLMInterface
llm = LLMInterface()
print(llm.test_connection())
"
```

#### Permission Issues
```bash
# Check directory permissions
python -c "
from hybrid_search_rag.config import DATA_DIR
import os
print(f'Data dir writable: {os.access(DATA_DIR, os.W_OK)}')
"
```

#### Model Access Issues
```bash
# Test model endpoints
python -c "
from hybrid_search_rag.embedding_services.gemini_embedder import GeminiEmbedder
embedder = GeminiEmbedder()
print(embedder.test_connection())
"
```

### Configuration Reset

To reset configuration to defaults:

```bash
# Backup current config
cp .env .env.backup

# Reset to default template
cp .env.example .env

# Re-run configuration
python -c "from hybrid_search_rag.config import run_config_checks; run_config_checks()"
```

## 📚 Next Steps

Once your configuration is complete:

1. **[Run your first query](./first-query.md)** - Test the system
2. **[Explore CLI commands](../interfaces/cli.md)** - Learn the command-line interface
3. **[Understand the architecture](../architecture/overview.md)** - Deep dive into how it works
4. **[Optimize performance](../performance/optimization.md)** - Scale for your needs

---

**Pro Tip**: Keep your `.env` file secure and never commit it to version control. Use `.env.example` as a template for others to configure their own environment.
