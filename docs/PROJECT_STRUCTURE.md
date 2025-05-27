# StudyAssistant Project Structure

## Overview
This document describes the organized structure of the StudyAssistant project after reorganization.

## Root Directory Structure

```
StudyAssistant/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── packages.txt             # System packages (for dev containers)
├── app.py                   # Streamlit web interface
├── .env                     # Environment variables (API keys)
├── .gitignore              # Git ignore rules
├── 
├── archive/                 # Legacy/backup files
│   ├── combining_dataset_papers.py
│   └── data_manager_legacy.py
│
├── data/                    # Main data storage
│   ├── __init__.py
│   ├── combined_metadata.json    # Document metadata
│   ├── combined_embeddings.npy   # Vector embeddings
│   ├── bm25_index.pkl            # BM25 search index
│   └── datasets/                 # Raw datasets
│       └── arxiv/
│
├── data_store/              # Additional data storage
│   └── evaluation_sets/     # Evaluation datasets
│
├── docs/                    # Documentation
│
├── tests/                   # Test files
│   ├── test_document_id_retrieval.py
│   ├── test_document_management.py
│   └── test_url_document_retrieval.py
│
├── scripts/                 # Command-line scripts
│   ├── __init__.py
│   ├── cli.py              # Main CLI interface
│   ├── continuous_fetch_with_dedup.py  # Document storage utilities
│   ├── bootstrap_evaluation_set.py     # Evaluation setup
│   ├── interactive_evaluation_labeler.py
│   ├── score_review_candidates.py
│   ├── data_collection/     # Data collection scripts
│   │   ├── arxiv_parallel_dataset_retrieval.py
│   │   └── combine_dataset.py
│   ├── data_management/     # Data management utilities
│   │   ├── document_integration.py
│   │   ├── document_retrieval.py
│   │   ├── document_search.py
│   │   └── update_metadata.py
│   └── tools/              # Utility tools
│       ├── common_utils.py
│       └── fix_js_escapes.py
│
├── hybrid_search_rag/       # Core library package
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── data_handling/      # Data processing and storage
│   │   ├── __init__.py
│   │   ├── data_manager.py      # Data persistence layer
│   │   └── resource_fetcher.py  # External data fetching
│   ├── embedding_services/ # Text embedding services
│   │   ├── __init__.py
│   │   ├── gemini_embedder.py   # Google Gemini embeddings
│   │   └── manual_embedder.py   # Manual/local embeddings
│   ├── evaluation/         # Evaluation framework
│   │   ├── __init__.py
│   │   ├── evaluator.py         # Main evaluation logic
│   │   └── metrics.py           # Evaluation metrics
│   ├── llm_services/       # LLM integration
│   │   ├── __init__.py
│   │   └── llm_interface.py     # LLM API interface
│   ├── retrieval_algorithm/ # Search and retrieval
│   │   ├── __init__.py
│   │   └── hybrid_recommender.py  # Hybrid search implementation
│   ├── text_processing/    # Text processing utilities
│   │   ├── __init__.py
│   │   └── nltk_processor.py    # NLTK-based processing
│   └── utils/              # General utilities
│       └── __init__.py
│
├── finetune/               # Model fine-tuning components
├── logs/                   # Application logs
├── my_mcp_server/         # MCP server components
└── nltk_data/             # NLTK data cache
```

## Key Components

### Scripts Directory (`scripts/`)
- **`cli.py`**: Main command-line interface for the application
- **`continuous_fetch_with_dedup.py`**: Document storage and deduplication utilities
- **Evaluation scripts**: Tools for creating and managing evaluation datasets
- **Data management**: Tools for document search, retrieval, and integration

### Core Library (`hybrid_search_rag/`)
- **`config.py`**: Central configuration management
- **`data_handling/`**: Data persistence and external data fetching
- **`embedding_services/`**: Text embedding generation
- **`llm_services/`**: Large Language Model integration
- **`retrieval_algorithm/`**: Hybrid search implementation
- **`evaluation/`**: Evaluation framework and metrics

### Data Storage
- **`data/`**: Primary data storage for processed documents, embeddings, and indices
- **`data_store/`**: Additional storage for evaluation datasets
- **`archive/`**: Legacy files and backups

## Import Structure

The project follows a hierarchical import structure:
- Scripts import from the `hybrid_search_rag` package
- Data management scripts use relative imports within the `scripts` package
- Core library modules use relative imports within their package structure

## Usage

1. **Main CLI**: `python scripts/cli.py [command] [options]`
2. **Web Interface**: `python app.py` (Streamlit)
3. **Data Collection**: Scripts in `scripts/data_collection/`
4. **Evaluation**: Scripts for evaluation setup and labeling

## Configuration

All configuration is centralized in `hybrid_search_rag/config.py`, including:
- API keys (loaded from environment variables)
- File paths and storage locations
- Model configurations
- Processing parameters
