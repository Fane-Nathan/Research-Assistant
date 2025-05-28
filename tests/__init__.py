"""
Test suite for the Research Assistant application.

This directory contains all test files organized by category:
- unit/: Unit tests for individual components
- rag_tests/: Integration tests for RAG functionality
- deployment verification scripts

Test Categories:
1. Unit Tests: Testing individual functions and classes
2. RAG Tests: End-to-end retrieval and generation testing
3. Deployment Tests: Verification scripts for cloud deployment

Running Tests:
- Individual tests: python -m pytest tests/unit/test_specific.py
- All unit tests: python -m pytest tests/unit/
- RAG tests: python -m pytest tests/rag_tests/
- Deployment verification: python deployment_verification.py

Test Dependencies:
- pytest for unit testing framework
- asyncio for async test support
- All project dependencies from requirements.txt
"""