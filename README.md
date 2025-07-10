# Research Assistant 🔍

> A powerful AI-powered research assistant that combines hybrid search with Retrieval-Augmented Generation (RAG) to help you find, understand, and synthesize academic content.

## 🌐 Live Demo

**Try it now:** [Research Assistant](https://research-assistant-flx.streamlit.app/)

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
┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │    │   Processing    │     │   Retrieval     │
│                 │    │                 │     │                 │
│ • arXiv API     │───▶│ • Text Chunking │───▶│ • Vector Search │
│ • Web Crawling  │    │ • Embeddings    │     │ • BM25 Search   │
│ • Custom Docs   │    │ • Indexing      │     │ • Hybrid Fusion │
└─────────────────┘    └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐               │
│  User Interface │    │   Generation    │               │
│                 │    │                 │               │
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
