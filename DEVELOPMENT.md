# Development Guide

Welcome to the Research Assistant development guide! This document will help you set up your development environment and contribute to the project.

## 🚀 Quick Start for Developers

### Prerequisites

- Python 3.9 or higher
- Git
- Google Gemini API key
- Node.js 18+ (for documentation)

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Research-Assistant.git
   cd Research-Assistant
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # For development tools
   pip install -r development/requirements_development.txt
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Verify Installation**
   ```bash
   python -c "from hybrid_search_rag import ResearchAssistant; print('Setup successful!')"
   ```

## 📁 Project Structure

```
Research-Assistant/
├── app.py                 # Streamlit web interface
├── hybrid_search_rag/     # Core library
│   ├── __init__.py
│   ├── core/             # Core functionality
│   ├── models/           # Data models
│   ├── services/         # Business logic
│   └── utils/            # Utilities
├── scripts/              # CLI tools
├── tests/                # Test suite
├── docs/                 # Documentation (Docusaurus)
├── development/          # Development tools and analysis
├── data/                 # Data storage (gitignored)
└── .github/              # GitHub workflows and templates
```

## 🧪 Testing

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=hybrid_search_rag --cov-report=html
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names
- Mock external dependencies
- Test both success and failure cases

### Test Guidelines

```python
def test_search_functionality():
    """Test that search returns relevant results."""
    # Arrange
    assistant = ResearchAssistant()
    
    # Act
    results = assistant.search("machine learning")
    
    # Assert
    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)
```

## 🎨 Code Style

We use automated formatting and linting:

```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type checking
mypy hybrid_search_rag/
```

### Style Guidelines

- **Line Length**: 88 characters (Black default)
- **Imports**: Use isort for import organization
- **Docstrings**: Google style docstrings
- **Type Hints**: Use type hints for public functions
- **Naming**: Use snake_case for functions and variables

## 🏗️ Architecture

### Core Components

1. **ResearchAssistant**: Main interface class
2. **HybridRetriever**: Combines vector and keyword search
3. **DocumentProcessor**: Handles document chunking and embedding
4. **RAGProcessor**: Manages retrieval-augmented generation

### Key Patterns

- **Dependency Injection**: Services are injected rather than hardcoded
- **Factory Pattern**: For creating different types of retrievers
- **Strategy Pattern**: For different search strategies
- **Observer Pattern**: For progress tracking

## 📝 Documentation

### Building Documentation

```bash
cd docs
npm install
npm run start  # Development server
npm run build  # Production build
```

### Documentation Standards

- **API Reference**: Auto-generated from docstrings
- **Tutorials**: Step-by-step guides
- **Examples**: Practical use cases
- **Architecture**: High-level design documents

## 🔄 Git Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/feature-name`: New features
- `bugfix/bug-name`: Bug fixes
- `hotfix/issue-name`: Critical fixes

### Commit Messages

Follow conventional commits:

```
feat: add new search ranking algorithm
fix: resolve issue with document chunking
docs: update API documentation
test: add unit tests for retrieval
refactor: improve code organization
```

### Pull Request Process

1. Create feature branch from `develop`
2. Make changes and add tests
3. Ensure all tests pass
4. Update documentation if needed
5. Create PR with clear description
6. Address review feedback
7. Merge after approval

## 🚀 Deployment

### Local Development

```bash
# Web interface
streamlit run app.py

# CLI tool
python scripts/cli.py --help
```

### Production Deployment

- **Streamlit Cloud**: Automated deployment from GitHub
- **Docker**: Containerized deployment
- **GitHub Actions**: CI/CD pipeline

## 🐛 Debugging

### Common Issues

1. **Import Errors**: Check Python path and virtual environment
2. **API Key Issues**: Verify environment variables
3. **Memory Issues**: Reduce batch sizes for large documents
4. **Performance**: Use profiling tools to identify bottlenecks

### Debugging Tools

```bash
# Profile performance
python -m cProfile -o profile.stats app.py

# Memory profiling
python -m memory_profiler app.py

# Debug logging
export LOG_LEVEL=DEBUG
python app.py
```

## 📊 Performance

### Optimization Guidelines

- **Batch Processing**: Process documents in batches
- **Caching**: Cache embeddings and search results
- **Async Operations**: Use async for I/O operations
- **Memory Management**: Clear unused variables

### Monitoring

- **Response Times**: Track API response times
- **Memory Usage**: Monitor memory consumption
- **Error Rates**: Track and alert on errors
- **User Metrics**: Monitor user interactions

## 🤝 Contributing

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Check our comprehensive docs

### Review Process

1. **Code Review**: All changes require peer review
2. **Automated Tests**: CI/CD pipeline must pass
3. **Manual Testing**: Test functionality manually
4. **Documentation**: Update docs for user-facing changes

## 📚 Resources

- **Python Best Practices**: [PEP 8](https://pep8.org/)
- **Testing**: [pytest documentation](https://docs.pytest.org/)
- **Git**: [Conventional Commits](https://conventionalcommits.org/)
- **Documentation**: [Docusaurus](https://docusaurus.io/)

---

Happy coding! 🎉
