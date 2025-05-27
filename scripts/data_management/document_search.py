#!/usr/bin/env python3
"""
Script for searching and retrieving documents from the document storage.
Provides advanced search capabilities and document retrieval options.
"""

import os
import sys
import json
import gzip
import argparse
import logging
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
import time

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

# Define default document storage directory
DEFAULT_DOCUMENT_STORAGE_DIR = os.path.join(project_root, "data_hybrid", "documents")
DOCUMENT_HASHES_FILE = os.path.join(project_root, "data_hybrid", "document_hashes.json")

class DocumentRepository:
    """Class for managing document search and retrieval"""
    
    def __init__(self, storage_dir=DEFAULT_DOCUMENT_STORAGE_DIR):
        self.storage_dir = storage_dir
        self.document_index = None
        self.document_hashes = {}
        self.topics = set()
        
        # Check if storage directory exists
        if not os.path.exists(self.storage_dir):
            logger.warning(f"Document storage directory does not exist: {self.storage_dir}")
        else:
            logger.info(f"Using document storage directory: {self.storage_dir}")
            
        # Load document hashes if available
        self._load_document_hashes()
    
    def _load_document_hashes(self):
        """Load document hashes from file"""
        if os.path.exists(DOCUMENT_HASHES_FILE):
            try:
                with open(DOCUMENT_HASHES_FILE, 'r', encoding='utf-8') as f:
                    self.document_hashes = json.load(f)
                logger.info(f"Loaded {len(self.document_hashes)} document hashes")
            except Exception as e:
                logger.error(f"Failed to load document hashes: {e}")
    
    def build_index(self, refresh=False):
        """
        Build or refresh the document index.
        This creates an in-memory index of all documents for faster searching.
        """
        if self.document_index is not None and not refresh:
            return self.document_index
            
        logger.info("Building document index...")
        start_time = time.time()
        
        self.document_index = []
        self.topics = set()
        
        try:
            # Walk through the storage directory
            for dirpath, dirnames, filenames in os.walk(self.storage_dir):
                # Extract topic from path
                rel_path = os.path.relpath(dirpath, self.storage_dir)
                if rel_path == '.':
                    continue
                
                topic = rel_path.split(os.path.sep)[0] if os.path.sep in rel_path else rel_path
                self.topics.add(topic)
                
                # Process files
                for filename in filenames:
                    if not (filename.endswith('.json') or filename.endswith('.json.gz') or 
                            filename.endswith('.txt') or filename.endswith('.txt.gz')):
                        continue
                    
                    file_path = os.path.join(dirpath, filename)
                    
                    # Extract metadata without loading full content
                    try:
                        doc_id = None
                        doc_title = None
                        doc_source = None
                        doc_hash = None
                        doc_type = 'json' if '.json' in filename else 'txt'
                        doc_compressed = filename.endswith('.gz')
                        
                        # For JSON files, extract metadata
                        if doc_type == 'json':
                            if doc_compressed:
                                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                    metadata = json.load(f)
                            else:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                    
                            doc_id = metadata.get('id', os.path.splitext(filename)[0])
                            doc_title = metadata.get('title', 'Untitled')
                            doc_source = metadata.get('source', 'unknown')
                            doc_published = metadata.get('published', '')
                            doc_authors = metadata.get('authors', [])
                        else:
                            # For TXT files, extract from filename
                            doc_id = os.path.splitext(filename)[0]
                            if doc_compressed:
                                doc_id = os.path.splitext(doc_id)[0]  # Remove .gz
                            doc_title = f"Document {doc_id}"
                            doc_source = "unknown"
                            doc_published = ""
                            doc_authors = []
                            
                            # Try to read header info from text file
                            try:
                                if doc_compressed:
                                    opener = gzip.open(file_path, 'rt', encoding='utf-8')
                                else:
                                    opener = open(file_path, 'r', encoding='utf-8')
                                    
                                with opener as f:
                                    # Read first few lines for header info
                                    header_lines = [f.readline() for _ in range(10) if f.readline()]
                                    for line in header_lines:
                                        if line.startswith("Title:"):
                                            doc_title = line[6:].strip()
                                        elif line.startswith("Authors:"):
                                            authors_str = line[8:].strip()
                                            doc_authors = [a.strip() for a in authors_str.split(',')]
                                        elif line.startswith("Published:"):
                                            doc_published = line[10:].strip()
                                        elif line.startswith("Source:"):
                                            doc_source = line[7:].strip()
                                        elif line.startswith("ID:"):
                                            doc_id = line[3:].strip()
                            except Exception as e:
                                logger.debug(f"Could not read text file header: {e}")
                        
                        # Get file stats
                        file_stats = os.stat(file_path)
                        file_size = file_stats.st_size
                        file_modified = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                        
                        # Get document hash from hash registry if available
                        if doc_id in self.document_hashes:
                            doc_hash = self.document_hashes[doc_id]
                        
                        # Add to index
                        self.document_index.append({
                            'id': doc_id,
                            'title': doc_title,
                            'source': doc_source,
                            'topic': topic,
                            'authors': doc_authors if 'doc_authors' in locals() else [],
                            'published': doc_published if 'doc_published' in locals() else '',
                            'hash': doc_hash,
                            'path': file_path,
                            'type': doc_type,
                            'compressed': doc_compressed,
                            'size': file_size,
                            'modified': file_modified
                        })
                        
                    except Exception as e:
                        logger.error(f"Error indexing document {file_path}: {e}")
            
            # Sort index by modified date (newest first)
            self.document_index.sort(key=lambda x: x['modified'], reverse=True)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Indexed {len(self.document_index)} documents across {len(self.topics)} topics in {elapsed_time:.2f} seconds")
            
            return self.document_index
        except Exception as e:
            logger.error(f"Error building document index: {e}")
            return []
    
    def search(self, query=None, doc_id=None, topic=None, source=None, format=None, 
               compressed=None, limit=100, include_content=False):
        """
        Search for documents in the repository.
        Returns a list of matching documents.
        """
        # Ensure index is built
        if self.document_index is None:
            self.build_index()
            
        # If no index could be built, return empty list
        if not self.document_index:
            return []
            
        results = []
        
        # Start with all documents
        candidates = self.document_index.copy()
        
        # Filter by document ID
        if doc_id:
            # Normalize document IDs for comparison
            normalized_id = self.normalize_id(doc_id)
            candidates = [
                doc for doc in candidates if 
                self.normalize_id(doc['id']) == normalized_id or
                normalized_id in self.normalize_id(doc['id']) or 
                self.normalize_id(doc['id']) in normalized_id
            ]
            
        # Filter by topic
        if topic:
            candidates = [doc for doc in candidates if topic.lower() in doc['topic'].lower()]
            
        # Filter by source
        if source:
            candidates = [doc for doc in candidates if source.lower() in doc['source'].lower()]
            
        # Filter by format
        if format:
            candidates = [doc for doc in candidates if doc['type'] == format]
            
        # Filter by compression
        if compressed is not None:
            candidates = [doc for doc in candidates if doc['compressed'] == compressed]
            
        # If a query is provided, search in title and content
        if query:
            query = query.lower()
            matching_docs = []
            
            # First check titles (faster)
            for doc in candidates:
                if query in doc['title'].lower():
                    matching_docs.append(doc)
                    
            # For remaining documents, if content is requested, check content
            if include_content:
                for doc in candidates:
                    if doc in matching_docs:
                        continue
                        
                    try:
                        content = self.get_document_content(doc['path'], doc['compressed'])
                        if query in content.lower():
                            matching_docs.append(doc)
                    except Exception as e:
                        logger.error(f"Error searching document content {doc['path']}: {e}")
                        
                candidates = matching_docs
            else:
                # If we're not checking content, just use title matches
                candidates = matching_docs
        
        # Apply limit if needed
        results = candidates[:limit] if limit > 0 else candidates
        
        # Include content if requested
        if include_content and results:
            for doc in results:
                try:
                    if 'path' in doc:
                        doc['content'] = self.get_document_content(doc['path'], doc['compressed'])
                except Exception as e:
                    logger.error(f"Error retrieving document content: {e}")
                    doc['content'] = f"Error: {e}"
                    
        # Remove path from results (usually not needed by callers)
        for doc in results:
            if 'path' in doc:
                doc.pop('path', None)
            
        return results
    
    def get_document_content(self, file_path, compressed=False):
        """Retrieve document content from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        try:
            # JSON files
            if file_path.endswith('.json') or file_path.endswith('.json.gz'):
                if compressed:
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                return data.get('content', '')
            
            # Text files
            else:
                if compressed:
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        content = f.read()
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Skip header lines if present
                lines = content.split('\n')
                header_end = 0
                for i, line in enumerate(lines[:10]):
                    if line.startswith(('Title:', 'Authors:', 'Published:', 'Source:', 'ID:')):
                        header_end = i + 1
                    elif i > header_end and line.strip():
                        break
                
                return '\n'.join(lines[header_end:])
        
        except Exception as e:
            raise RuntimeError(f"Error reading document {file_path}: {e}")
    
    def get_document_by_id(self, doc_id):
        """Retrieve a specific document by ID with its content"""
        # First try an exact search
        results = self.search(doc_id=doc_id, limit=1, include_content=True)
        if results:
            return results[0]
            
        # Generate alternative versions of the ID to try
        alternative_ids = []
        
        # If it has underscores, try replacing them with slashes or colons
        if '_' in doc_id:
            alternative_ids.extend([
                doc_id.replace('_', '/'),
                doc_id.replace('_', ':'),
                doc_id.replace('_', '/').replace('_', ':')
            ])
        
        # If it has slashes, try replacing them with underscores
        if '/' in doc_id:
            alternative_ids.append(doc_id.replace('/', '_'))
            
        # If it has colons, try replacing them with underscores or URL encoding
        if ':' in doc_id:
            alternative_ids.extend([
                doc_id.replace(':', '_'),
                doc_id.replace(':', '%3A')
            ])
            
        # If it's a URL, try to extract just the path part
        if doc_id.startswith('http'):
            try:
                import urllib.parse
                parsed_url = urllib.parse.urlparse(doc_id)
                path_id = parsed_url.path
                if path_id.startswith('/'):
                    path_id = path_id[1:]  # Remove leading slash
                
                alternative_ids.extend([
                    path_id,
                    path_id.replace('/', '_').replace(':', '_')
                ])
            except:
                pass
                
        # Try all alternative IDs
        for alt_id in alternative_ids:
            results = self.search(doc_id=alt_id, limit=1, include_content=True)
            if results:
                return results[0]
                
        # If no exact match is found, use the normalized ID to check for partial matches
        normalized_id = self.normalize_id(doc_id)
        if self.document_index:
            for doc in self.document_index:
                doc_normalized = self.normalize_id(doc['id'])
                if normalized_id in doc_normalized or doc_normalized in normalized_id:
                    return self.search(doc_id=doc['id'], limit=1, include_content=True)[0]
                
        return None
    
    def get_topics(self):
        """Get a list of all available topics"""
        if self.document_index is None:
            self.build_index()
        return sorted(list(self.topics))
    
    def get_stats(self):
        """Get statistics about the document repository"""
        if self.document_index is None:
            self.build_index()
            
        try:
            # If we couldn't build an index or it's empty
            if not self.document_index:
                return {
                    'total_documents': 0,
                    'total_topics': 0,
                    'storage_size_bytes': 0,
                    'storage_size_human': '0 bytes',
                    'compressed_documents': 0,
                    'sources': {},
                    'formats': {'json': 0, 'txt': 0}
                }
                
            stats = {
                'total_documents': len(self.document_index),
                'total_topics': len(self.topics),
                'storage_size_bytes': 0,
                'compressed_documents': 0,
                'sources': {},
                'formats': {
                    'json': 0,
                    'txt': 0
                }
            }
            
            # Collect statistics
            for doc in self.document_index:
                stats['storage_size_bytes'] += doc['size']
                
                if doc['compressed']:
                    stats['compressed_documents'] += 1
                    
                stats['formats'][doc['type']] += 1
                
                source = doc['source']
                if source not in stats['sources']:
                    stats['sources'][source] = 0
                stats['sources'][source] += 1
                
            # Add human-readable size
            stats['storage_size_human'] = format_size(stats['storage_size_bytes'])
            
            return stats
        
        except Exception as e:
            logger.error(f"Error calculating repository stats: {e}")
            return {
                'total_documents': len(self.document_index) if self.document_index else 0,
                'total_topics': len(self.topics) if hasattr(self, 'topics') and self.topics else 0,
                'storage_size_bytes': 0,
                'storage_size_human': '0 bytes',
                'compressed_documents': 0,
                'sources': {},
                'formats': {'json': 0, 'txt': 0},
                'error': str(e)
            }
    
    def export_documents(self, output_file, query=None, topic=None, limit=0, format='json'):
        """
        Export documents to a single file.
        Returns the number of documents exported.
        """
        # Search for documents to export
        documents = self.search(query=query, topic=topic, limit=limit, include_content=True)
        
        if not documents:
            logger.warning("No documents found to export")
            return 0
            
        try:
            # Decide format
            if format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(documents, f, indent=2, ensure_ascii=False)
            
            elif format == 'txt':
                with open(output_file, 'w', encoding='utf-8') as f:
                    for doc in documents:
                        f.write(f"Document ID: {doc['id']}\n")
                        f.write(f"Title: {doc['title']}\n")
                        f.write(f"Topic: {doc['topic']}\n")
                        f.write(f"Source: {doc['source']}\n")
                        f.write(f"Authors: {', '.join(doc['authors'])}\n")
                        f.write(f"Published: {doc['published']}\n")
                        f.write("\n" + "="*80 + "\n\n")
                        f.write(doc['content'])
                        f.write("\n\n" + "="*80 + "\n\n")
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            logger.info(f"Exported {len(documents)} documents to {output_file}")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error exporting documents: {e}")
            return 0

    def normalize_id(self, doc_id):
        """Normalize document ID for consistent comparison"""
        if not doc_id:
            return ""
            
        normalized = doc_id.lower()
        
        # Handle common URL-encoded characters
        normalized = normalized.replace('%3a', ':').replace('%3A', ':')
        normalized = normalized.replace('%2f', '/').replace('%2F', '/')
        
        # If it's a full URL, extract just the path
        if normalized.startswith('http'):
            try:
                import urllib.parse
                parsed_url = urllib.parse.urlparse(normalized)
                path = parsed_url.path
                if path.startswith('/'):
                    path = path[1:]  # Remove leading slash
                normalized = path
            except:
                # If parsing fails, keep the original
                pass
                
        return normalized

def format_size(size_bytes):
    """Format a size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"

def print_document_info(doc, show_content=False, max_content_length=1000):
    """Print formatted document information"""
    print(f"\n{'='*80}")
    print(f"Document ID: {doc['id']}")
    print(f"Title: {doc['title']}")
    print(f"Topic: {doc['topic']}")
    print(f"Source: {doc['source']}")
    
    if 'authors' in doc and doc['authors']:
        print(f"Authors: {', '.join(doc['authors'])}")
    
    if 'published' in doc and doc['published']:
        print(f"Published: {doc['published']}")
    
    if 'size' in doc:
        print(f"Size: {format_size(doc['size'])}")
    
    if 'modified' in doc:
        print(f"Last Modified: {doc['modified']}")
    
    if show_content and 'content' in doc:
        print(f"\n{'-'*80}\nContent:")
        content = doc['content']
        if max_content_length > 0 and len(content) > max_content_length:
            print(f"{content[:max_content_length]}...\n[Content truncated, total length: {len(content)} characters]")
        else:
            print(content)
    
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Search and retrieve documents from the document storage")
    parser.add_argument("--storage-dir", type=str, default=DEFAULT_DOCUMENT_STORAGE_DIR,
                        help=f"Path to document storage directory (default: {DEFAULT_DOCUMENT_STORAGE_DIR})")
    
    # Search options
    parser.add_argument("--query", type=str, help="Search query in document titles and content")
    parser.add_argument("--id", type=str, help="Search for a specific document ID")
    parser.add_argument("--topic", type=str, help="Filter by topic")
    parser.add_argument("--source", type=str, help="Filter by source (e.g., arxiv)")
    
    # Display options
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return (default: 10, 0 for unlimited)")
    parser.add_argument("--show-content", action="store_true", help="Show document content in results")
    parser.add_argument("--content-length", type=int, default=1000, help="Maximum content length to display (default: 1000, 0 for unlimited)")
    
    # Actions
    parser.add_argument("--list-topics", action="store_true", help="List all available topics")
    parser.add_argument("--stats", action="store_true", help="Show statistics about the document repository")
    parser.add_argument("--export", type=str, help="Export documents to a file (specify output file path)")
    parser.add_argument("--export-format", type=str, choices=["json", "txt"], default="json", help="Format for export (default: json)")
    parser.add_argument("--refresh-index", action="store_true", help="Force refresh of the document index")
    
    args = parser.parse_args()
    
    # Create document repository
    repo = DocumentRepository(storage_dir=args.storage_dir)
    
    # Force refresh index if requested
    if args.refresh_index:
        repo.build_index(refresh=True)
    
    # List topics if requested
    if args.list_topics:
        topics = repo.get_topics()
        print(f"\nAvailable Topics ({len(topics)}):")
        for topic in topics:
            print(f"  - {topic}")
        print()
        return 0
    
    # Show stats if requested
    if args.stats:
        stats = repo.get_stats()
        print("\nDocument Repository Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Total Topics: {stats['total_topics']}")
        print(f"  Storage Size: {stats['storage_size_human']}")
        print(f"  Compressed Documents: {stats['compressed_documents']} ({stats['compressed_documents']/stats['total_documents']*100:.1f}%)")
        
        print("\n  Documents by Format:")
        for format, count in stats['formats'].items():
            print(f"    - {format}: {count} ({count/stats['total_documents']*100:.1f}%)")
        
        print("\n  Documents by Source:")
        for source, count in stats['sources'].items():
            print(f"    - {source}: {count} ({count/stats['total_documents']*100:.1f}%)")
        
        print()
        return 0
    
    # Export documents if requested
    if args.export:
        count = repo.export_documents(
            args.export,
            query=args.query,
            topic=args.topic,
            limit=args.limit,
            format=args.export_format
        )
        if count > 0:
            print(f"Successfully exported {count} documents to {args.export}")
        else:
            print("No documents were exported")
        return 0
    
    # Search for documents
    results = repo.search(
        query=args.query,
        doc_id=args.id,
        topic=args.topic,
        source=args.source,
        limit=args.limit,
        include_content=args.show_content
    )
    
    # Display results
    if not results:
        print("No documents found matching your criteria")
        return 0
    
    print(f"\nFound {len(results)} document(s):")
    
    for i, doc in enumerate(results):
        print_document_info(doc, args.show_content, args.content_length)
        
        # Pagination for large result sets
        if i < len(results) - 1 and (i + 1) % 3 == 0 and len(results) > 3:
            if input(f"\nShowing {i+1} of {len(results)} results. Press Enter to continue, or 'q' to quit: ").lower() == 'q':
                break
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
