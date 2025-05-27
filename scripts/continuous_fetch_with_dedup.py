# -*- coding: utf-8 -*-
"""
Continuous fetching with deduplication functionality for StudyAssistant.

Provides document saving functionality and storage directory configuration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Get project root - go up from scripts/ directory to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DOCUMENT_STORAGE_DIR = os.path.join(project_root, "data", "documents")


def save_document(
    document_data: Dict[str, Any],
    storage_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> bool:
    """
    Save a document to the storage directory.
    
    Args:
        document_data: Document metadata and content to save
        storage_dir: Directory to save the document (uses default if None)
        filename: Custom filename (auto-generated if None)
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Use default storage directory if not provided
        if storage_dir is None:
            storage_dir = DEFAULT_DOCUMENT_STORAGE_DIR
            
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            doc_id = document_data.get('id', 'unknown')
            filename = f"{doc_id}.json"
            
        file_path = os.path.join(storage_dir, filename)
        
        # Save document as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Document saved successfully: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save document: {e}", exc_info=True)
        return False


def load_document(document_id: str, storage_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load a document from the storage directory.
    
    Args:
        document_id: ID of the document to load
        storage_dir: Directory to load from (uses default if None)
        
    Returns:
        Document data if found, None otherwise
    """
    try:
        if storage_dir is None:
            storage_dir = DEFAULT_DOCUMENT_STORAGE_DIR
            
        file_path = os.path.join(storage_dir, f"{document_id}.json")
        
        if not os.path.exists(file_path):
            logger.warning(f"Document not found: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            document_data = json.load(f)
            
        logger.info(f"Document loaded successfully: {file_path}")
        return document_data
        
    except Exception as e:
        logger.error(f"Failed to load document {document_id}: {e}", exc_info=True)
        return None


def document_exists(document_id: str, storage_dir: Optional[str] = None) -> bool:
    """
    Check if a document exists in the storage directory.
    
    Args:
        document_id: ID of the document to check
        storage_dir: Directory to check (uses default if None)
        
    Returns:
        True if document exists, False otherwise
    """
    if storage_dir is None:
        storage_dir = DEFAULT_DOCUMENT_STORAGE_DIR
        
    file_path = os.path.join(storage_dir, f"{document_id}.json")
    return os.path.exists(file_path)


def list_documents(storage_dir: Optional[str] = None) -> list:
    """
    List all documents in the storage directory.
    
    Args:
        storage_dir: Directory to list (uses default if None)
        
    Returns:
        List of document IDs
    """
    if storage_dir is None:
        storage_dir = DEFAULT_DOCUMENT_STORAGE_DIR
        
    if not os.path.exists(storage_dir):
        return []
        
    document_ids = []
    for filename in os.listdir(storage_dir):
        if filename.endswith('.json'):
            document_id = filename[:-5]  # Remove .json extension
            document_ids.append(document_id)
            
    return document_ids


if __name__ == "__main__":
    # Example usage
    sample_doc = {
        "id": "test_doc_001",
        "title": "Test Document",
        "content": "This is a test document for the StudyAssistant project.",
        "source": "manual",
        "timestamp": "2025-05-27"
    }
    
    # Save test document
    success = save_document(sample_doc)
    if success:
        print(f"Test document saved to: {DEFAULT_DOCUMENT_STORAGE_DIR}")
        
        # List documents
        docs = list_documents()
        print(f"Documents in storage: {docs}")
        
        # Load test document
        loaded_doc = load_document("test_doc_001")
        if loaded_doc:
            print(f"Loaded document: {loaded_doc['title']}")
