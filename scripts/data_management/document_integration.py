#!/usr/bin/env python3
"""
Document integration utilities for the StudyAssistant project.
This script provides tools for integrating document storage with
the rest of the application's data management.
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set

from scripts.common_utils import calculate_document_hash # MODIFIED: Import from common_utils

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
        get_repository, find_documents, get_document, 
        get_topics, get_stats
    )
    from hybrid_search_rag import config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class DocumentIntegrator:
    """
    Class for integrating document storage with the rest of the application.
    Provides tools for synchronizing documents with the combined_metadata.json file.
    """
    
    def __init__(self, metadata_file_path: str, document_storage_base_dir: str): # MODIFIED: Updated signature
        self.repository = get_repository()
        
        self.metadata_file = metadata_file_path # MODIFIED: Use passed argument
        self.document_storage_base_dir = document_storage_base_dir # MODIFIED: Store passed argument
        self.backup_file = f"{self.metadata_file}.bak.{int(time.time())}"
        
        # For file locking in integrate method
        import threading
        self.lock = threading.Lock()
        
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Loads metadata from the metadata file."""
        if not os.path.exists(self.metadata_file):
            logger.warning(f"Metadata file {self.metadata_file} does not exist. Returning empty list.")
            return []
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content: # Handle empty file
                    logger.warning(f"Metadata file {self.metadata_file} is empty. Returning empty list.")
                    return []
                metadata = json.loads(content)
            if not isinstance(metadata, list):
                logger.warning(f"Metadata file {self.metadata_file} does not contain a JSON list. Found {type(metadata)}. Returning empty list.")
                return []
            logger.info(f"Loaded {len(metadata)} documents from {self.metadata_file}")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from metadata file {self.metadata_file}: {e}. Returning empty list.")
            # Optionally, attempt to read backup or handle corruption
            return []
        except Exception as e:
            logger.error(f"Error loading metadata file {self.metadata_file}: {e}. Returning empty list.")
            return []

    def _save_metadata(self, metadata: List[Dict[str, Any]]):
        """Saves metadata to the metadata file and creates a backup."""
        if not isinstance(metadata, list):
            logger.error(f"Attempted to save non-list data to metadata file: {type(metadata)}. Aborting save.")
            return

        try:
            # Create backup of existing file before overwriting
            if os.path.exists(self.metadata_file):
                # Ensure the original file is not empty and is valid JSON before backing up
                try:
                    with open(self.metadata_file, 'r', encoding='utf-8') as f_orig_check:
                        original_content = f_orig_check.read().strip()
                        if original_content: # Only backup if there's content
                            json.loads(original_content) # Validate JSON
                            # Now proceed with backup
                            with open(self.backup_file, 'w', encoding='utf-8') as f_bak:
                                f_bak.write(original_content) # Write the original content as is
                            logger.info(f"Created backup at {self.backup_file}")
                        else:
                            logger.info(f"Original metadata file {self.metadata_file} is empty. Skipping backup.")
                except json.JSONDecodeError:
                    logger.warning(f"Original metadata file {self.metadata_file} is corrupted. Skipping backup.")
                except Exception as backup_read_exc:
                    logger.error(f"Error reading original metadata for backup: {backup_read_exc}. Skipping backup.")


            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved {len(metadata)} documents to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata file: {e}")

    def update_metadata_from_documents(self, topic=None):
        """
        Update the combined_metadata.json file with documents from storage.
        This allows documents to be used by the search engine and other components.
        
        Args:
            topic (str, optional): Only update documents from this topic
            
        Returns:
            tuple: (documents_added, documents_updated, total_documents)
        """
        # Load existing metadata using the robust _load_metadata method
        metadata = self._load_metadata() 
        
        # Get existing document IDs
        existing_ids = set()
        id_to_index = {}
        for i, doc in enumerate(metadata): # metadata is now guaranteed to be a list
            doc_id = doc.get('entry_id', '')
            if doc_id:
                existing_ids.add(doc_id)
                id_to_index[doc_id] = i
        
        # Find documents in storage
        documents = find_documents(topic=topic, limit=0, include_content=True)
        logger.info(f"Found {len(documents)} documents in storage{' for topic ' + topic if topic else ''}")
        
        # Track changes
        documents_added = 0
        documents_updated = 0
        
        # Process each document
        for doc in documents:
            doc_id = doc.get('id', '')
            if not doc_id:
                logger.warning(f"Skipping document with missing ID: {doc.get('title', 'N/A')}")
                continue
            
            # Create document metadata entry
            doc_metadata = {
                'entry_id': doc_id,
                'original_title': doc.get('title', ''),
                'original_url': doc_id, # Assuming doc_id is the URL or a unique identifier that can serve as one
                'authors': doc.get('authors', []),
                'published': doc.get('published', ''),
                'source': doc.get('source', 'arxiv'), # Default to arxiv, adjust if other sources are used
                'chunk_id': f"{doc_id}_chunk_0", # Assuming single chunk for now
                'chunk_text': doc.get('content', ''),
                'chunk_num': 0,
                'total_chunks': 1,
                'storage_path': doc.get('path', ''),
                'last_updated': datetime.now().isoformat()
            }
            
            # Add or update metadata
            if doc_id in existing_ids:
                # Update existing entry
                metadata[id_to_index[doc_id]] = doc_metadata
                documents_updated += 1
            else:
                # Add new entry
                metadata.append(doc_metadata)
                documents_added += 1
        
        # Save updated metadata using the robust _save_metadata method
        self._save_metadata(metadata)
        
        if documents_added > 0 or documents_updated > 0:
            logger.info(f"Added {documents_added} new documents, updated {documents_updated} existing documents.")
        else:
            logger.info("No new documents added or existing documents updated in metadata.")
            
        return documents_added, documents_updated, len(metadata)
    
    def synchronize_metadata_with_documents(self, source_identifier: Optional[str] = None) -> Tuple[int, int, int, int]: # MODIFIED: Added source_identifier and return type
        """
        Make sure all documents in metadata are available in document storage,
        and all documents in storage are represented in the metadata.
        Updates metadata if discrepancies are found for documents existing in storage but not metadata.
        
        Args:
            source_identifier (Optional[str]): The specific topic (slug) to synchronize. 
                                               If None, synchronizes based on all documents found in storage.

        Returns:
            tuple: (final_metadata_docs_count, final_storage_docs_count, 
                    docs_in_metadata_but_missing_in_storage_count, 
                    docs_initially_in_storage_but_missing_in_metadata_count)
        """
        initial_metadata_list = self._load_metadata()
        # Ensure metadata_file path is robustly checked or handled by _load_metadata
        # if not initial_metadata_list and not os.path.exists(self.metadata_file):
        #     logger.warning(f"Metadata file {self.metadata_file} does not exist. Cannot synchronize effectively.")
            # return 0, 0, 0, 0 # Or handle as appropriate

        initial_metadata_ids = {doc.get('entry_id') for doc in initial_metadata_list if doc.get('entry_id')}
        
        storage_documents = find_documents(limit=0) # Get all documents from storage
        storage_ids = {doc['id'] for doc in storage_documents if 'id' in doc}
        
        missing_in_storage_ids = initial_metadata_ids - storage_ids
        missing_in_metadata_ids = storage_ids - initial_metadata_ids
        
        logger.info(f"Initial Sync Check: Metadata has {len(initial_metadata_ids)} docs. Storage has {len(storage_ids)} docs.")
        if missing_in_storage_ids:
            logger.warning(f"Initial Sync Check: {len(missing_in_storage_ids)} docs in metadata are MISSING from storage.")
        if missing_in_metadata_ids:
            logger.info(f"Initial Sync Check: {len(missing_in_metadata_ids)} docs in storage are MISSING from metadata. Will attempt to add them.")

        final_metadata_count_after_sync = len(initial_metadata_ids)

        if missing_in_metadata_ids:
            logger.info(f"Updating metadata with {len(missing_in_metadata_ids)} documents found in storage but not in metadata.")
            # update_metadata_from_documents processes all topics if topic=None
            # If source_identifier is provided, it will be used as the topic filter.
            added_count, updated_count, total_in_metadata_file = self.update_metadata_from_documents(topic=source_identifier)
            logger.info(f"Metadata update process complete: {added_count} added, {updated_count} updated. Metadata file now has {total_in_metadata_file} entries.")
            final_metadata_count_after_sync = total_in_metadata_file
        else:
            logger.info("No documents found in storage that were missing from metadata. Metadata is up-to-date with storage.")
            # If no update was run, the count is from the initial load.
            # We can reload to be absolutely sure, or trust total_in_metadata_file from update_metadata_from_documents
            # For simplicity, if update_metadata_from_documents was not called, we use the initial count.
            # If an update happened, total_in_metadata_file is the most accurate.
            # If you want to be absolutely certain after any operation, reload:
            # final_metadata_list = self._load_metadata()
            # final_metadata_count_after_sync = len(final_metadata_list)


        final_storage_count = len(storage_ids) # Storage count doesn't change in this method
        
        return final_metadata_count_after_sync, final_storage_count, len(missing_in_storage_ids), len(missing_in_metadata_ids)

    
    def extract_documents_from_metadata(self, reprocess=False):
        """
        Extract documents from the combined_metadata.json file and store them
        in the document storage system. This is useful for migrating existing 
        data to the new document storage format.
        
        Args:
            reprocess (bool): If True, reprocess all documents even if they already exist
            
        Returns:
            tuple: (documents_processed, documents_stored, documents_skipped)
        """
        # Import moved here to avoid issues if this class is imported elsewhere before continuous_fetch_with_dedup is fully defined
        from ..continuous_fetch_with_dedup import save_document, DEFAULT_DOCUMENT_STORAGE_DIR
        
        if not os.path.exists(self.metadata_file):
            logger.error(f"Metadata file {self.metadata_file} does not exist")
            return 0, 0, 0
            
        try:
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Group chunks by original document
            documents = {}
            for chunk in metadata:
                doc_id = chunk.get('entry_id', '')
                original_url = chunk.get('original_url', '')
                
                if not doc_id or not original_url:
                    continue
                
                # Create document entry or append chunk
                if original_url not in documents:
                    documents[original_url] = {
                        'id': doc_id,
                        'title': chunk.get('original_title', 'Untitled'),
                        'authors': chunk.get('authors', []),
                        'published': chunk.get('published', ''),
                        'source': chunk.get('source', 'unknown'),
                        'content': chunk.get('chunk_text', ''),
                        'topic': self._extract_topic_from_title(chunk.get('original_title', ''))
                    }
                else:
                    # Append chunk text if it's for the same document
                    documents[original_url]['content'] += "\n" + chunk.get('chunk_text', '')
            
            logger.info(f"Extracted {len(documents)} unique documents from metadata")
            
            # Import each document to storage
            from ..continuous_fetch_with_dedup import save_document
            
            documents_stored = 0
            documents_skipped = 0
            
            for url, doc_data in documents.items(): # Renamed doc to doc_data for clarity
                # Check if document already exists in storage
                existing = find_documents(doc_id=doc_data['id'], limit=1)
                
                if existing and not reprocess:
                    documents_skipped += 1
                    continue
                
                # Prepare arguments for save_document from continuous_fetch_with_dedup.py
                doc_content_to_save = doc_data.get('content', '')
                doc_id_to_save = doc_data['id']
                # Metadata for save_document should be a dict of relevant fields
                metadata_for_save_doc = {
                    'title': doc_data.get('title', ''),
                    'authors': doc_data.get('authors', []),
                    'abstract': doc_data.get('abstract', ''), # Assuming abstract might be in doc_data
                    'source': doc_data.get('source', 'unknown'),
                    'url': url # Original URL can be part of metadata
                }
                # Construct topic_dir using the stored base directory and the doc's topic
                # Fallback to DEFAULT_DOCUMENT_STORAGE_DIR if self.document_storage_base_dir is not set (e.g. direct instantiation)
                base_storage_dir = self.document_storage_base_dir or DEFAULT_DOCUMENT_STORAGE_DIR
                topic_dir_for_save = os.path.join(base_storage_dir, doc_data.get('topic', 'uncategorized'))

                # Call save_document with named arguments
                # progress_tracker and console are optional and can be None
                saved_status, _, _ = save_document(
                    doc_content=doc_content_to_save,
                    doc_id=doc_id_to_save,
                    metadata=metadata_for_save_doc,
                    topic_dir=topic_dir_for_save,
                    progress_tracker=None, 
                    console=None
                )
                                
                if saved_status:
                    documents_stored += 1
                else:
                    logger.warning(f"Failed to store document {doc_data['id']}")
            
            logger.info(f"Processed {len(documents)} documents: {documents_stored} stored, {documents_skipped} skipped")
            return len(documents), documents_stored, documents_skipped
            
        except Exception as e:
            logger.error(f"Error extracting documents from metadata: {e}")
            return 0, 0, 0
    
    def _extract_topic_from_title(self, title):
        """Extract a topic from the document title"""
        # Default topic if we can't determine one
        default_topic = "uncategorized"
        
        if not title:
            return default_topic
            
        # Get available topics
        topics = get_topics()
        
        # Check if title contains any of our topics
        title_lower = title.lower()
        for topic in topics:
            if topic.lower() in title_lower:
                return topic
                
        # Try to match with keywords
        keywords = {
            "machine learning": "machine learning optimization",
            "deep learning": "deep learning architectures",
            "neural network": "neural network attention mechanisms",
            "nlp": "natural language processing",
            "natural language": "natural language processing",
            "transformer": "transformer models",
            "llm": "large language models",
            "reinforcement learning": "reinforcement learning algorithms",
            "computer vision": "computer vision object detection",
            "gan": "generative adversarial networks",
            "federated": "federated learning",
            "blockchain": "blockchain technology",
            "quantum": "quantum computing algorithms",
            "security": "cybersecurity threat detection",
            "database": "database optimization",
            "cloud": "cloud computing architectures",
            "edge computing": "edge computing",
            "iot": "internet of things",
            "algorithm": "algorithmic complexity",
            "data visualization": "data visualization techniques",
            "anomaly detection": "anomaly detection",
            "time series": "time series forecasting",
            "bayesian": "bayesian inference",
            "causal": "causal inference",
            "recommender": "recommender systems",
            "clustering": "clustering algorithms",
            "robotics": "robotic motion planning",
            "autonomous": "autonomous vehicles",
            "human-robot": "human-robot interaction",
            "physics": "quantum mechanics",
            "relativity": "relativity theory",
            "particle physics": "particle physics",
            "astrophysics": "astrophysics",
            "condensed matter": "condensed matter physics",
            "optimization": "optimization theory",
            "graph theory": "graph theory",
            "topology": "topology",
            "differential equation": "differential equations",
            "numerical method": "numerical methods",
            "computational biology": "computational biology",
            "neuroscience": "neuroscience computing",
            "financial": "financial machine learning",
            "climate": "climate modeling",
            "healthcare": "healthcare analytics"
        }
        
        for keyword, topic in keywords.items():
            if keyword.lower() in title_lower:
                return topic
                
        return default_topic
    
    def integrate(self, document_data: Dict[str, Any]) -> bool:
        """
        Integrates a single document's metadata into the combined_metadata.json file.
        This method should handle loading, updating, and saving the metadata file
        with appropriate file locking to prevent corruption.
        """
        if not document_data or not document_data.get('id'):
            logger.warning("Integrate: Document data is empty or missing ID. Skipping.")
            return False

        doc_id = document_data['id']
        logger.debug(f"Integrate: Attempting to integrate document ID: {doc_id}")

        try:
            with self.lock: 
                metadata_list = self._load_metadata()
                
                existing_doc_index = -1
                # Ensure consistent key for matching (e.g. 'entry_id' or 'id')
                # The current code uses 'id' for matching, but creates 'entry_id'
                # Let's assume 'id' from document_data is the primary key to check against 'id' in metadata_list
                for i, meta_doc in enumerate(metadata_list):
                    if meta_doc.get('id') == doc_id: 
                        existing_doc_index = i
                        break
                
                new_meta_entry = {
                    "entry_id": doc_id, 
                    "id": doc_id, 
                    "title": document_data.get('title', 'Untitled'),
                    "authors": document_data.get('authors', []),
                    "published": document_data.get('published', ''),
                    "source": document_data.get('source', 'unknown'),
                    "topic": document_data.get('topic', 'unknown'),
                    "storage_path": document_data.get('storage_path'), 
                    "storage_timestamp": document_data.get('storage_timestamp', datetime.now().isoformat()),
                    "content_hash": calculate_document_hash( 
                        document_data.get('content',''), 
                        document_data.get('title',''), 
                        document_data.get('authors',[]),
                        document_data.get('abstract', '')
                    ) 
                }

                if existing_doc_index != -1:
                    logger.info(f"Integrate: Updating existing document in metadata: {doc_id}")
                    # Ensure all fields are updated, not just a subset if using dict.update()
                    metadata_list[existing_doc_index].update(new_meta_entry) 
                else:
                    logger.info(f"Integrate: Adding new document to metadata: {doc_id}")
                    metadata_list.append(new_meta_entry)
                
                self._save_metadata(metadata_list)
                logger.debug(f"Integrate: Successfully integrated document ID: {doc_id}")
                return True
        except Exception as e:
            logger.error(f"Integrate: Failed to integrate document {doc_id}: {e}", exc_info=True)
            return False

def main():
    """Main function for document integration utilities"""
    parser = argparse.ArgumentParser(description="Document integration utilities")
    parser.add_argument("--update-metadata", action="store_true", help="Update metadata file with documents from storage")
    parser.add_argument("--topic", type=str, help="Only process documents from this topic")
    parser.add_argument("--extract-documents", action="store_true", help="Extract documents from metadata and store them")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess documents even if they already exist")
    parser.add_argument("--synchronize", action="store_true", help="Synchronize metadata with documents")
    
    args = parser.parse_args()
    
    # MODIFIED: Instantiate DocumentIntegrator with required arguments
    # These paths should ideally come from a shared config or be dynamically determined
    # For now, using paths relative to project_root, assuming config.DATA_DIR is 'data_hybrid'
    # and config.METADATA_FILE is 'combined_metadata.json'
    # Ensure config.DATA_DIR and config.METADATA_FILE are correctly defined and accessible
    # Fallback to direct construction if config is not as expected.
    
    metadata_dir = os.path.join(project_root, "data_hybrid") # Default if config.DATA_DIR is not set
    metadata_filename = "combined_metadata.json" # Default if config.METADATA_FILE is not set
    
    if hasattr(config, 'DATA_DIR') and config.DATA_DIR:
        metadata_dir = config.DATA_DIR # Use config.DATA_DIR if available
        if not os.path.isabs(metadata_dir): # Ensure it's an absolute path
             metadata_dir = os.path.join(project_root, metadata_dir)
    
    if hasattr(config, 'METADATA_FILE') and config.METADATA_FILE:
        metadata_filename = config.METADATA_FILE

    metadata_file_path = os.path.join(metadata_dir, metadata_filename)
    
    # Assuming document storage base directory is also configurable or a known default
    # For example, from continuous_fetch_with_dedup.py, DEFAULT_DOCUMENT_STORAGE_DIR
    # This might need to be passed as an arg or read from config as well.
    # For now, let's use a path consistent with continuous_fetch_with_dedup.py's default.
    default_doc_storage_dir = os.path.join(project_root, "data_hybrid", "documents")
    document_storage_base_dir = getattr(config, 'DOCUMENT_STORAGE_DIR', default_doc_storage_dir)
    if not os.path.isabs(document_storage_base_dir):
        document_storage_base_dir = os.path.join(project_root, document_storage_base_dir)


    integrator = DocumentIntegrator(metadata_file_path, document_storage_base_dir)

    if args.update_metadata:
        integrator.update_metadata_from_documents(topic=args.topic)
    
    elif args.extract_documents:
        integrator.extract_documents_from_metadata(reprocess=args.reprocess)
    
    elif args.synchronize:
        logger.info("Starting metadata synchronization...")
        # Pass the source_identifier from argparse to the method
        meta_count, storage_count, missing_storage, missing_meta = integrator.synchronize_metadata_with_documents(source_identifier=args.source_identifier)
        logger.info(f"Synchronization Complete: Metadata Docs: {meta_count}, Storage Docs: {storage_count}")
        if missing_storage > 0:
            logger.warning(f"Docs in metadata but missing from storage: {missing_storage}")
        if missing_meta > 0:
            logger.info(f"Docs added to metadata from storage: {missing_meta}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
