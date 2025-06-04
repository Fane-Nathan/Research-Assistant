#!/usr/bin/env python3
"""
Production Repository Cleanup Script

This script cleans up the Research Assistant repository for production release:
- Removes development artifacts and temporary files
- Organizes project structure
- Cleans up documentation
- Prepares for public GitHub release
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List

def get_base_path():
    """Get the base project path."""
    return Path(__file__).parent

def remove_files_and_dirs(patterns: List[str], base_path: Path, description: str):
    """Remove files and directories matching patterns."""
    print(f"\n🧹 {description}")
    removed_count = 0
    
    for pattern in patterns:
        matches = list(base_path.glob(pattern))
        for match in matches:
            try:
                if match.is_file():
                    match.unlink()
                    print(f"  ✅ Removed file: {match.name}")
                elif match.is_dir():
                    shutil.rmtree(match)
                    print(f"  ✅ Removed directory: {match.name}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {match}: {e}")
    
    if removed_count == 0:
        print(f"  ℹ️  No items found matching patterns")
    else:
        print(f"  📊 Total items removed: {removed_count}")

def organize_development_files(base_path: Path):
    """Move development files to appropriate locations."""
    print(f"\n📁 Organizing development files")
    
    # Create development directory if it doesn't exist
    dev_dir = base_path / "development"
    dev_dir.mkdir(exist_ok=True)
    
    # Files to move to development directory
    dev_files = [
        "analyze_problematic_queries.py",
        "analyze_retrieval_components.py", 
        "comprehensive_query_preprocessing_evaluation.py",
        "create_arxiv_evaluation_dataset.py",
        "diagnose_rag.py",
        "diagnose_retrieval_performance.py",
        "evaluate_query_preprocessing_impact.py",
        "optimize_rrf_parameters.py",
        "quick_query_preprocessing_eval.py",
        "run_real_metrics_evaluation.py",
        "verify_optimal_preprocessing.py",
        "simple_verification.py"
    ]
    
    moved_count = 0
    for file_name in dev_files:
        src = base_path / file_name
        if src.exists():
            dst = dev_dir / file_name
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✅ Moved: {file_name} → development/")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Failed to move {file_name}: {e}")
    
    print(f"  📊 Total files moved: {moved_count}")

def create_gitignore(base_path: Path):
    """Create or update .gitignore file."""
    print(f"\n📝 Creating/updating .gitignore")
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# Research Assistant specific
data/
data_hybrid/
data_store/
logs/
nltk_data/
*.pkl
*.npy
*.log
evaluation_results_*.json
*_results.json
*_analysis.log
development/
archive/
finetune/
frontend/
my_mcp_server/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
*~
"""
    
    gitignore_path = base_path / ".gitignore"
    try:
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print(f"  ✅ Updated .gitignore")
    except Exception as e:
        print(f"  ❌ Failed to update .gitignore: {e}")

def create_production_readme(base_path: Path):
    """Create a production-ready README."""
    print(f"\n📖 Creating production README")
    
    readme_content = '''# Research Assistant 🔍

> A powerful AI-powered research assistant that combines hybrid search with Retrieval-Augmented Generation (RAG) to help you find, understand, and synthesize academic content.

## 🌐 Live Demo

**Try it now:** [https://research-assistant-flx.streamlit.app/](https://research-assistant-flx.streamlit.app/)

Experience the full functionality through our interactive web interface:
- 📚 Research paper search and analysis
- 🔍 Hybrid search capabilities (semantic + keyword)
- 🤖 Real-time RAG-powered question answering
- 📊 Document processing and knowledge base management

## ✨ Features

### 🎯 **Intelligent Search**
- **Hybrid Retrieval**: Combines semantic (vector) and keyword (BM25) search
- **Multi-Source**: arXiv papers, web articles, custom documents
- **Smart Ranking**: Advanced fusion algorithms for relevance

### 🧠 **AI-Powered Analysis**
- **RAG Integration**: Context-aware answers using retrieved documents
- **Multiple LLM Support**: Google Gemini, OpenAI, and more
- **Flexible Modes**: Strict RAG (document-only) or hybrid (document + general knowledge)

### 📊 **Data Processing**
- **Smart Chunking**: Sentence-boundary aware document segmentation
- **High-Quality Embeddings**: Google Gemini text embeddings
- **Efficient Indexing**: Optimized storage and retrieval

### 🖥️ **Multiple Interfaces**
- **Web App**: Beautiful Streamlit interface
- **CLI Tool**: Powerful command-line interface
- **Python API**: Programmatic access for integration

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Gemini API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Fane-Nathan/Research-Assistant.git
   cd Research-Assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the web interface**
   ```bash
   streamlit run app.py
   ```

## 📖 Documentation

- 📘 **[Complete Documentation](https://fane-nathan.github.io/Research-Assistant/)**
- 🚀 **[Quick Start Guide](docs/docs/getting-started/quick-start.md)**
- 🏗️ **[Architecture Overview](docs/docs/architecture/overview.md)**
- 🔧 **[Configuration Guide](docs/docs/getting-started/configuration.md)**

## 🛠️ Usage Examples

### Web Interface
Visit [our live demo](https://research-assistant-flx.streamlit.app/) or run locally:
```bash
streamlit run app.py
```

### Command Line
```bash
# Search arXiv papers
python scripts/cli.py search "machine learning transformers"

# Process custom documents
python scripts/cli.py process --url "https://example.com/paper.pdf"

# Ask questions about your documents
python scripts/cli.py query "What are the main contributions of this paper?"
```

### Python API
```python
from hybrid_search_rag import ResearchAssistant

# Initialize
assistant = ResearchAssistant()

# Search and get answers
results = assistant.search_and_answer(
    query="What are recent advances in neural networks?",
    mode="hybrid_rag"
)
```

## 🏗️ Architecture

Research Assistant is built with a modular, scalable architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │   Retrieval     │
│                 │    │                 │    │                 │
│ • arXiv API     │───▶│ • Text Chunking │───▶│ • Vector Search │
│ • Web Crawling  │    │ • Embeddings    │    │ • BM25 Search   │
│ • Custom Docs   │    │ • Indexing      │    │ • Hybrid Fusion │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐              │
│  User Interface │    │   Generation    │              │
│                 │    │                 │              │
│ • Web App       │◀───│ • LLM Reasoning │◀─────────────┘
│ • CLI Tool      │    │ • RAG Synthesis │
│ • Python API    │    │ • Answer Format │
└─────────────────┘    └─────────────────┘
```

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests  
python -m pytest tests/integration/

# Deployment verification
python deployment_verification.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [https://fane-nathan.github.io/Research-Assistant/](https://fane-nathan.github.io/Research-Assistant/)
- **Live Demo**: [https://research-assistant-flx.streamlit.app/](https://research-assistant-flx.streamlit.app/)
- **Issues**: [GitHub Issues](https://github.com/Fane-Nathan/Research-Assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Fane-Nathan/Research-Assistant/discussions)

## 🙏 Acknowledgments

- Google Gemini for embeddings and LLM capabilities
- The open-source community for amazing libraries
- arXiv for providing free access to research papers

---

<div align="center">

**Built with ❤️ for the research community**

[⭐ Star this repo](https://github.com/Fane-Nathan/Research-Assistant) | [🐛 Report Bug](https://github.com/Fane-Nathan/Research-Assistant/issues) | [💡 Request Feature](https://github.com/Fane-Nathan/Research-Assistant/issues)

</div>
'''
    
    readme_path = base_path / "README.md"
    try:
        # Backup current README
        if readme_path.exists():
            backup_path = base_path / "README_backup.md"
            shutil.copy2(readme_path, backup_path)
            print(f"  📋 Backed up current README to README_backup.md")
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"  ✅ Created production README.md")
    except Exception as e:
        print(f"  ❌ Failed to create README: {e}")

def main():
    """Main cleanup function."""
    print("🚀 Starting Research Assistant Production Cleanup")
    print("=" * 55)
    
    base_path = get_base_path()
    print(f"📁 Working directory: {base_path}")
    
    # 1. Remove development artifacts and temporary files
    temp_patterns = [
        "*.log",
        "*_results.json", 
        "*_analysis.log",
        "*.tmp",
        "*.temp",
        "__pycache__",
        "*.pyc",
        "app.log",
        "logs-*",
        "test_*.py",
        "final_*.py", 
        "deployment_verification*.py"
    ]
    remove_files_and_dirs(temp_patterns, base_path, "Removing temporary files and logs")
    
    # 2. Remove obsolete directories
    obsolete_dirs = [
        "frontend",
        "my_mcp_server", 
        "finetune"
    ]
    remove_files_and_dirs(obsolete_dirs, base_path, "Removing obsolete directories")
    
    # 3. Organize development files
    organize_development_files(base_path)
    
    # 4. Update .gitignore
    create_gitignore(base_path)
    
    # 5. Create production README
    create_production_readme(base_path)
    
    print("\n" + "=" * 55)
    print("✅ Production cleanup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Review the new README.md")
    print("   2. Check development/ directory for moved files")
    print("   3. Verify .gitignore covers all necessary patterns")
    print("   4. Run tests to ensure functionality is preserved")
    print("   5. Commit changes and push to GitHub")
    print("\n🎉 Your repository is now production-ready!")

if __name__ == "__main__":
    main()
