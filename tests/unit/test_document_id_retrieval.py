#!/usr/bin/env python3
"""
Script to test document retrieval with different URL formats
"""

import sys
import os
import argparse
import logging

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

# Import document repository
try:
    from scripts.document_search import DocumentRepository
    from scripts.document_retrieval import get_document
except ImportError as e:
    logger.error(f"Failed to import document modules: {e}")
    sys.exit(1)

def test_document_retrieval(doc_id):
    """Test document retrieval with different format variations"""
    print(f"\n=== Testing document retrieval for ID: {doc_id} ===\n")
    
    # Create variations of the document ID
    variations = [
        doc_id,
        doc_id.replace('/', '_').replace(':', '_'),  # Safe ID for URLs
        doc_id.replace('/', '_'),  # Just replace slashes
        doc_id.replace(':', '_'),  # Just replace colons
        doc_id.replace(':', '%3A'),  # URL encode colons
        doc_id.replace('/', '%2F'),  # URL encode slashes
    ]
    
    # If it starts with http://, create additional variations
    if doc_id.startswith(('http://', 'https://')):
        # Import URL parsing
        try:
            import urllib.parse
            parsed_url = urllib.parse.urlparse(doc_id)
            path_id = parsed_url.path
            if path_id.startswith('/'):
                path_id = path_id[1:]  # Remove leading slash
                
            variations.extend([
                path_id,  # Just the path component
                path_id.replace('/', '_').replace(':', '_'),  # Path with safe encoding
            ])
        except Exception as e:
            logger.error(f"Error parsing URL: {e}")
    
    # Try to retrieve document with each variation
    for i, var_id in enumerate(variations):
        print(f"Variation {i+1}: '{var_id}'")
        document = get_document(var_id)
        
        if document:
            print(f"✅ SUCCESS - Retrieved document:")
            print(f"  Title: {document.get('title', 'No title')}")
            print(f"  ID in document: {document.get('id', 'No ID')}")
            print(f"  Topic: {document.get('topic', 'No topic')}")
            print()
        else:
            print(f"❌ FAILED - No document found with this ID variation\n")
    
    # Create a repository and use normalize_id
    repo = DocumentRepository()
    if hasattr(repo, 'normalize_id'):
        print("\n=== Testing ID normalization ===\n")
        for i, var_id in enumerate(variations):
            normalized_id = repo.normalize_id(var_id)
            print(f"Variation {i+1}: '{var_id}' -> Normalized: '{normalized_id}'")

def main():
    parser = argparse.ArgumentParser(description="Test document retrieval with different ID formats")
    parser.add_argument("doc_id", help="Document ID to test retrieval with")
    args = parser.parse_args()
    
    # Test document retrieval
    test_document_retrieval(args.doc_id)

if __name__ == "__main__":
    main()
