#!/usr/bin/env python3
"""
Document retrieval module for the StudyAssistant project.
This module provides a simplified API for retrieving documents
that can be used by other parts of the application.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(name)s - %(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

try:
    from .document_search import DocumentRepository, DEFAULT_DOCUMENT_STORAGE_DIR
except ImportError as e:
    logger.error(f"Failed to import DocumentRepository: {e}")
    sys.exit(1)

_repository = None

def get_repository(storage_dir=DEFAULT_DOCUMENT_STORAGE_DIR, refresh=False):
    """Get or create the document repository instance"""
    global _repository
    
    if _repository is None:
        _repository = DocumentRepository(storage_dir=storage_dir)
        _repository.build_index()
    elif refresh:
        _repository.build_index(refresh=True)
        
    return _repository

def find_documents(query=None, doc_id=None, topic=None, limit=10, include_content=True):
    """
    Find documents matching the specified criteria.
    
    Args:
        query (str, optional): Text to search for in document titles and content
        doc_id (str, optional): Document ID to search for
        topic (str, optional): Topic to filter by
        limit (int, optional): Maximum number of results to return
        include_content (bool, optional): Whether to include document content in results
        
    Returns:
        List[Dict[str, Any]]: List of matching documents
    """
    repo = get_repository()
    return repo.search(
        query=query,
        doc_id=doc_id,
        topic=topic,
        limit=limit,
        include_content=include_content
    )

def get_document(doc_id):
    """
    Get a specific document by ID.
    
    Args:
        doc_id (str): ID of the document to retrieve
        
    Returns:
        Dict[str, Any] or None: The document if found, None otherwise
    """
    repo = get_repository()
    return repo.get_document_by_id(doc_id)

def get_topics():
    """
    Get a list of all available topics.
    
    Returns:
        List[str]: List of topic names
    """
    repo = get_repository()
    return repo.get_topics()

def get_stats():
    """
    Get statistics about the document repository.
    
    Returns:
        Dict[str, Any]: Statistics about the repository
    """
    repo = get_repository()
    return repo.get_stats()

def export_documents(output_file, query=None, topic=None, limit=0, format='json'):
    """
    Export documents to a file.
    
    Args:
        output_file (str): Path to the output file
        query (str, optional): Text to search for in document titles and content
        topic (str, optional): Topic to filter by
        limit (int, optional): Maximum number of results to export (0 for unlimited)
        format (str, optional): Export format ('json' or 'txt')
        
    Returns:
        int: Number of documents exported
    """
    repo = get_repository()
    return repo.export_documents(
        output_file=output_file,
        query=query,
        topic=topic,
        limit=limit,
        format=format
    )

# Example usage
if __name__ == "__main__":
    print("\nDocument Repository API Example:")
    
    topics = get_topics()
    print(f"Available Topics ({len(topics)}): {', '.join(topics[:5])}...")
    
    stats = get_stats()
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Storage Size: {stats.get('storage_size_human', '0 bytes')}")
    
    if topics:
        first_topic = topics[0]
        docs = find_documents(topic=first_topic, limit=2)
        
        print(f"\nSample Documents from '{first_topic}':")
        for doc in docs:
            print(f"  - {doc['title']} (ID: {doc['id']})")
            if 'content' in doc:
                content_preview = doc['content'][:100].replace('\n', ' ')
                print(f"    Content Preview: {content_preview}...")
    
    print("\nFor more advanced search options, use the document_search.py script directly.")
