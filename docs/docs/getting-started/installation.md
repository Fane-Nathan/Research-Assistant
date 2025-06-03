# Installation Guide

This guide will walk you through setting up the Research Assistant on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Git** for cloning the repository
- **Node.js 18+** (for Docusaurus documentation, optional)

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 2GB+ free space for dependencies and data
- **Network**: Internet connection for API access and data fetching

## Installation Methods

### Method 1: Clone from GitHub (Recommended)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/research-assistant.git
   cd research-assistant
   ```

2. **Set Up Python Environment**
   
   Using **conda** (recommended):
   ```bash
   conda create -n research_env python=3.11 -y
   conda activate research_env
   ```
   
   Using **venv**:
   ```bash
   python -m venv research_env
   # Windows
   research_env\Scripts\activate
   # macOS/Linux
   source research_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # Development dependencies (optional)
   pip install -r requirements_development.txt
   
   # Streamlit web app dependencies (optional)
   pip install -r requirements_streamlit.txt
   ```

4. **Install System Dependencies**
   
   **For Playwright (web scraping)**:
   ```bash
   playwright install
   ```
   
   **For NLTK (text processing)**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

### Method 2: Docker Installation (Coming Soon)

Docker support is planned for future releases to simplify deployment.

## Post-Installation Setup

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp .env.example .env
```

Edit the `.env` file with your API keys:

```env
# Google Gemini API (for embeddings and LLM)
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API (alternative LLM provider)
OPENAI_API_KEY=your_openai_api_key_here

# Groq API (alternative LLM provider)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Custom model endpoints
CUSTOM_LLM_ENDPOINT=https://your-custom-endpoint.com
```

### 2. Verify Installation

Run the configuration check:

```bash
python -c "from hybrid_search_rag.config import run_config_checks; run_config_checks()"
```

This should output something like:
```
✅ Project root detected: /path/to/research-assistant
✅ Environment variables loaded
✅ API keys configured
✅ Data directories created
```

### 3. Initial Data Setup

Create necessary directories:

```bash
python -c "
from hybrid_search_rag.config import DATA_DIR
import os
os.makedirs(DATA_DIR, exist_ok=True)
print(f'Data directory created: {DATA_DIR}')
"
```

## Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version
# Should be 3.10 or higher
```

#### Package Installation Failures

If you encounter issues with package installation:

1. **Update pip**:
   ```bash
   pip install --upgrade pip
   ```

2. **Install packages individually**:
   ```bash
   pip install playwright aiohttp numpy rank-bm25
   ```

3. **Use conda for scientific packages**:
   ```bash
   conda install numpy scipy scikit-learn
   ```

#### Playwright Installation Issues

If Playwright browser installation fails:

```bash
# Install system dependencies first
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libdrm2 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libasound2

# Then install browsers
playwright install
```

#### NLTK Data Download Issues

If NLTK data download fails:

```bash
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
"
```

#### Memory Issues

For large datasets, you may need to:

1. **Increase virtual memory** (swap space)
2. **Use a machine with more RAM**
3. **Process data in smaller batches**

### Getting Help

If you're still having trouble:

1. Check the [Troubleshooting Guide](../reference/troubleshooting.md)
2. Search [existing issues](https://github.com/your-org/research-assistant/issues)
3. Create a [new issue](https://github.com/your-org/research-assistant/issues/new) with:
   - Your operating system
   - Python version
   - Full error message
   - Steps to reproduce

## Next Steps

Once installation is complete:

1. 📖 [Configure your API keys](./configuration.md)
2. 🚀 [Run your first query](./first-query.md)
3. 💻 [Explore the CLI interface](../interfaces/cli.md)
4. 🌐 [Try the web interface](../interfaces/streamlit-app.md)

---

**Congratulations!** 🎉 You've successfully installed Research Assistant. You're ready to start exploring the world of AI-powered research assistance.
