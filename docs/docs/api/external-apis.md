# External APIs

Research Assistant integrates with several external APIs to provide comprehensive research functionality. This document outlines the APIs used and their configuration.

## Overview

The Research Assistant leverages multiple external services to provide powerful research capabilities:

- **arXiv API** - Academic paper search and retrieval
- **Large Language Model APIs** - AI-powered text generation and analysis
- **Embedding Services** - Vector embeddings for semantic search

## arXiv API

### Purpose
The arXiv API is used to search and retrieve academic papers from the arXiv repository.

### Configuration
- **Base URL**: `http://export.arxiv.org/api/query`
- **Rate Limiting**: Respects arXiv's usage guidelines
- **Authentication**: No API key required

### Usage
```python
# Example arXiv query
query = "machine learning transformers"
results = resource_fetcher.fetch_arxiv_papers(query, max_results=10)
```

### Features
- Search by keywords, categories, or authors
- Retrieve paper metadata (title, abstract, authors, etc.)
- Download PDF content for full-text analysis
- Automatic deduplication based on arXiv IDs

## Large Language Model APIs

### Supported Providers

#### Google Gemini
- **Models**: `gemini-1.5-flash-latest`, `gemini-pro`
- **Use Cases**: Text generation, question answering, analysis
- **Configuration**: Requires `GOOGLE_API_KEY` environment variable

#### Groq
- **Models**: Various high-performance models
- **Use Cases**: Fast inference for real-time applications
- **Configuration**: Requires `GROQ_API_KEY` environment variable

#### DeepSeek
- **Models**: DeepSeek language models
- **Use Cases**: Alternative LLM provider for fallback
- **Configuration**: Requires `DEEPSEEK_API_KEY` environment variable

### API Fallback Strategy

The system implements automatic fallback across providers:

1. **Primary**: Google Gemini
2. **Secondary**: Groq
3. **Tertiary**: DeepSeek

If one provider fails, the system automatically tries the next available provider.

### Rate Limiting and Error Handling

- Automatic retry with exponential backoff
- Rate limit detection and handling
- Graceful degradation when APIs are unavailable

## Embedding Services

### Google Gemini Embeddings
- **Model**: `models/embedding-001`
- **Dimensions**: 768-dimensional vectors
- **Use Case**: Semantic search and document similarity

### Configuration
```python
# In config.py
EMBEDDING_MODEL_NAME = 'models/embedding-001'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
```

## Security Considerations

### API Key Management
- All API keys are stored as environment variables
- Keys are never committed to version control
- Support for `.env` file configuration

### Rate Limiting
- Respectful usage of all external APIs
- Automatic rate limiting to prevent service disruption
- Monitoring and logging of API usage

### Error Handling
- Comprehensive error handling for all API calls
- Graceful degradation when services are unavailable
- Detailed logging for debugging and monitoring

## Usage Guidelines

### Best Practices
1. **Set up API keys** before using the system
2. **Monitor usage** to stay within provider limits
3. **Use caching** to reduce unnecessary API calls
4. **Implement retries** for transient failures

### Cost Optimization
- Cache embedding results to avoid recomputation
- Use appropriate model sizes for different tasks
- Monitor API usage and costs regularly

## Troubleshooting

### Common Issues

#### Missing API Keys
```
Error: GOOGLE_API_KEY environment variable not set
```
**Solution**: Set the required environment variables in your `.env` file.

#### Rate Limiting
```
Error: API rate limit exceeded
```
**Solution**: The system automatically handles rate limits with backoff strategies.

#### Network Issues
```
Error: Failed to connect to API endpoint
```
**Solution**: Check your internet connection and API endpoint availability.

## Development and Testing

### Local Development
- Use development API keys separate from production
- Test with smaller datasets to minimize API usage
- Use mock services for unit testing

### Production Deployment
- Secure API key storage (e.g., environment variables in cloud platforms)
- Monitor API usage and set up alerts
- Implement proper logging and error tracking