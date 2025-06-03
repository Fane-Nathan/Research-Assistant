# Streamlit Web Interface

The Research Assistant includes a modern, interactive web interface built with Streamlit that provides all the functionality of the CLI tool in an easy-to-use web application.

## 🌐 Live Demo

**Try it now:** [https://research-assistant-flx.streamlit.app/](https://research-assistant-flx.streamlit.app/)

The live demo is fully functional and includes:
- Real-time research paper search and retrieval
- Interactive hybrid search capabilities
- AI-powered question answering with RAG
- Document processing and knowledge base management
- Multi-provider LLM support with automatic fallback

## Features

### 🔍 Research Interface
- **arXiv Search**: Search and retrieve academic papers directly from arXiv
- **Web Crawling**: Extract content from research websites and documents
- **Document Upload**: Process your own PDF documents and research papers

### 🤖 AI-Powered Analysis
- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search
- **RAG Question Answering**: Get intelligent answers based on your knowledge base
- **Multi-LLM Support**: Automatic fallback across Google Gemini, Groq, and DeepSeek

### 📊 Data Management
- **Knowledge Base**: View and manage your document collection
- **Real-time Processing**: See documents being processed and indexed
- **Metadata Viewing**: Explore document metadata and embeddings

## Interface Components

### Search and Query
The main interface allows you to:
- Enter research queries for arXiv papers
- Specify web URLs for content extraction
- Ask questions about your knowledge base
- Choose between different RAG modes (strict vs. hybrid)

### Document Viewer
- Browse your processed document collection
- View document metadata and source information
- See embedding statistics and indexing status

### Settings and Configuration
- Configure LLM providers and API keys
- Adjust search parameters and retrieval settings
- Manage data directories and storage options

## Technical Architecture

### Frontend
- **Framework**: Streamlit for rapid prototyping and deployment
- **Styling**: Custom CSS for modern, responsive design
- **State Management**: Streamlit session state for user interactions

### Backend Integration
- **Async Processing**: Efficient handling of long-running operations
- **Resource Management**: Optimized for cloud deployment
- **Error Handling**: Graceful fallbacks and user-friendly error messages

### Cloud Deployment
- **Platform**: Streamlit Cloud
- **Dependencies**: Lightweight, optimized package selection
- **Performance**: Fast startup and responsive interactions

## Local Development

To run the Streamlit interface locally:

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Set up environment variables
# Create .env file with API keys:
# GOOGLE_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
# DEEPSEEK_API_KEY=your_key_here

# Run the application
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Deployment

For deployment instructions, see:
- [Streamlit Deployment Guide](https://github.com/Fane-Nathan/Research-Assistant/blob/alpha-release/STREAMLIT_DEPLOYMENT.md)
- [Deployment Status](https://github.com/Fane-Nathan/Research-Assistant/blob/alpha-release/DEPLOYMENT_STATUS.md)

## Comparison with CLI

| Feature | Web Interface | CLI Tool |
|---------|---------------|----------|
| **Ease of Use** | Intuitive GUI | Command-line expertise required |
| **Interactivity** | Real-time feedback | Batch processing |
| **Visualization** | Rich data display | Text-based output |
| **Deployment** | Cloud-hosted | Local installation |
| **Performance** | Optimized for web | Full feature set |
| **Customization** | Standard interface | Highly configurable |

## Troubleshooting

### Common Issues

**"Module not found" errors:**
- Ensure all dependencies are installed
- Check that the virtual environment is activated

**API key errors:**
- Verify API keys are set in Streamlit secrets (cloud) or .env file (local)
- Check that all required providers have valid keys

**Slow performance:**
- Large document collections may take time to process
- Consider using smaller batch sizes for initial testing

**Memory issues:**
- The web interface uses optimized dependencies
- For large-scale processing, consider using the CLI tool

### Getting Help

- Check the [troubleshooting guide](../reference/troubleshooting.md)
- Review [configuration documentation](../getting-started/configuration.md)
- Submit issues on the project repository