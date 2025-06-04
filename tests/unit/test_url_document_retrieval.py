#!/usr/bin/env python3
"""
Test script to ensure document retrieval works with URL-based document IDs.
"""

import os
import sys
import argparse

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.document_retrieval import get_document

def main():
    parser = argparse.ArgumentParser(description="Test document retrieval by URL")
    parser.add_argument("--doc-url", default="http://arxiv.org/abs/2504.17901v1", 
                       help="URL of the document to retrieve")
    args = parser.parse_args()
    
    doc_url = args.doc_url
    print(f"Testing retrieval of document by URL: {doc_url}")
    
    # 1. Direct lookup
    print("Method 1: Direct URL lookup")
    document = get_document(doc_url)
    if document:
        print(f"✅ SUCCESS - Found document: {document.get('title')}")
    else:
        print("❌ FAILED - Document not found")
    
    # 2. Path-based lookup
    print("\nMethod 2: Path-only lookup")
    import urllib.parse
    parsed_url = urllib.parse.urlparse(doc_url)
    path_id = parsed_url.path
    if path_id.startswith('/'):
        path_id = path_id[1:]  # Remove leading slash
    
    document = get_document(path_id)
    if document:
        print(f"✅ SUCCESS - Found document: {document.get('title')}")
    else:
        print("❌ FAILED - Document not found")
    
    # 3. Safe ID lookup
    print("\nMethod 3: Safe ID lookup")
    safe_id = doc_url.replace('/', '_').replace(':', '_')
    document = get_document(safe_id)
    if document:
        print(f"✅ SUCCESS - Found document: {document.get('title')}")
    else:
        print("❌ FAILED - Document not found")
    
    print("\nTest complete.")

if __name__ == "__main__":
    main()
