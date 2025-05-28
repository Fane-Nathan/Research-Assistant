# Streamlit Cloud Deployment Guide

## 🚀 Quick Start

### Step 1: Repository Setup
1. Push your code to GitHub (ensure all files are committed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository

### Step 2: App Configuration
Configure your Streamlit Cloud app with these **exact** settings:

```yaml
Main file path: app.py
Python version: 3.11
Requirements file: requirements_streamlit.txt
Packages file: packages_streamlit.txt
```

### Step 3: Environment Variables (Secrets)
In the Streamlit Cloud secrets section, add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here" 
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
```

## 📋 Deployment Checklist

### ✅ Pre-Deployment Verification
- [x] **Playwright warnings fixed**: App runs without sync API warnings
- [x] **Streamlined dependencies**: Using `requirements_streamlit.txt` (lightweight)
- [x] **System packages minimized**: Using `packages_streamlit.txt` (no Playwright deps)
- [x] **DeepSeek support**: Added `deepseek>=0.1.0` dependency
- [x] **No hardcoded paths**: App uses relative paths only
- [x] **Async-compatible**: ResourceFetcher properly handles async contexts

### 🔧 Key Technical Details
- **PDF Processing**: Automatic fallback to aiohttp (Playwright disabled in cloud)
- **Memory Optimization**: Heavy ML libraries removed for cloud deployment
- **Error Handling**: Graceful degradation when components unavailable
- **API Fallbacks**: Multi-provider LLM support (Google → Groq → DeepSeek)

### 📁 Important Files
- `app.py` - Main Streamlit application
- `requirements_streamlit.txt` - Lightweight Python dependencies  
- `packages_streamlit.txt` - Minimal system dependencies
- `hybrid_search_rag/` - Core RAG functionality package

### 🚨 Troubleshooting Common Issues

#### "Module not found" errors:
- Ensure all imports use relative paths
- Check that `hybrid_search_rag` package structure is maintained

#### "Playwright sync API" warnings:
- These are now fixed in ResourceFetcher
- App automatically detects async context and uses aiohttp

#### High memory usage/timeout:
- Verify using `requirements_streamlit.txt` (not `requirements.txt`)
- Heavy dependencies like `torch`, `tensorflow` are excluded

#### API key issues:
- Set secrets in Streamlit Cloud dashboard
- Use exact variable names: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`

## 🎯 Expected Deployment Result

Once deployed successfully, your app will provide:
- **Research Assistant Interface**: Full RAG functionality
- **arXiv Search**: Direct paper search and retrieval
- **Data Management**: Ability to update knowledge base
- **Multi-Provider LLM**: Automatic fallback across API providers
- **PDF Processing**: Via aiohttp (lightweight alternative to Playwright)
