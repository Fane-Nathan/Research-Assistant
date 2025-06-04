#!/usr/bin/env python3
"""
Test script for document search and integration.
Uses the new document management features to test document search, retrieval,
and metadata integration.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(name)s - %(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Import document retrieval API
try:
    from scripts.document_retrieval import (
        get_repository, find_documents, get_document, get_topics, get_stats
    )
    from scripts.document_integration import DocumentIntegrator
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def test_document_search():
    """Test document search functionality"""
    logger.info("Testing document search...")
    
    # Get repository
    repo = get_repository()
    
    # Get available topics
    topics = get_topics()
    logger.info(f"Available topics: {', '.join(topics)}")
    
    # Get stats
    stats = get_stats()
    logger.info(f"Document stats: {stats['total_documents']} documents, {stats['storage_size_human']}")
    
    # Search for documents
    if topics:
        topic = topics[0]
        logger.info(f"Searching for documents in topic '{topic}'...")
        results = find_documents(topic=topic, limit=5)
        
        logger.info(f"Found {len(results)} documents:")
        for i, doc in enumerate(results):
            logger.info(f"  {i+1}. {doc['title']} (ID: {doc['id']})")
            
        # Get full document content
        if results:
            doc_id = results[0]['id']
            logger.info(f"Retrieving full document for ID '{doc_id}'...")
            document = get_document(doc_id)
            
            if document and 'content' in document:
                content_preview = document['content'][:200].replace('\n', ' ')
                logger.info(f"Document content preview: {content_preview}...")
    else:
        logger.warning("No topics available for testing")

def test_document_integration():
    """Test document integration functionality"""
    logger.info("Testing document integration...")
    
    # Create integrator with required parameters
    metadata_file_path = os.path.join(project_root, "data_hybrid", "combined_metadata.json")
    document_storage_dir = os.path.join(project_root, "data_hybrid", "documents")
    integrator = DocumentIntegrator(metadata_file_path, document_storage_dir)
    
    # Synchronize metadata with documents
    logger.info("Synchronizing metadata with documents...")
    metadata_docs, storage_docs, missing_in_storage, missing_in_metadata = integrator.synchronize_metadata_with_documents()
    
    logger.info(f"Synchronization results:")
    logger.info(f"  Documents in metadata: {metadata_docs}")
    logger.info(f"  Documents in storage: {storage_docs}")
    logger.info(f"  Missing in storage: {missing_in_storage}")
    logger.info(f"  Missing in metadata: {missing_in_metadata}")
    
    # Update metadata from documents
    if storage_docs > 0:
        logger.info("Updating metadata from documents...")
        added, updated, total = integrator.update_metadata_from_documents()
        logger.info(f"  Documents added: {added}")
        logger.info(f"  Documents updated: {updated}")
        logger.info(f"  Total documents in metadata: {total}")

def main():
    """Main function to run tests"""
    logger.info("Starting document management tests")
    
    # Run document search tests
    test_document_search()
    
    # Run document integration tests
    test_document_integration()
    
    logger.info("Document management tests completed")

if __name__ == "__main__":
    main()
