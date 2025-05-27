# hybrid_search_rag/data/data_manager.py
"""Handles saving and loading project data: metadata, embeddings, and BM25 index."""

import os
import json
import numpy as np
import pickle # For object serialization (BM25 index)
import logging
from typing import List, Dict, Any, Tuple, Optional
import sys # For sys.path manipulation if needed, and error printing

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Attempt to import config from the user's project structure ---
# This assumes the script is run from a context where 'hybrid_search_rag' package is accessible
# (e.g., PROJECT_ROOT is in PYTHONPATH, or script is run from PROJECT_ROOT which contains hybrid_search_rag module)
try:
    # If your script is inside a package that also contains config, direct import might work.
    # If config is at a fixed relative path, you might need to adjust sys.path.
    # For example, if this script is in PROJECT_ROOT/scripts and config is in PROJECT_ROOT/hybrid_search_rag
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root_dir = os.path.dirname(current_dir) # Assuming script is one level down from project_root
    # sys.path.insert(0, project_root_dir)
    import config as config  # Attempt to import the user's config module   
    CONFIG_AVAILABLE = True
    logger.info("Successfully imported 'hybrid_search_rag.config'")
except ImportError as e:
    CONFIG_AVAILABLE = False
    logger.warning(f"Could not import 'hybrid_search_rag.config': {e}. "
                   "Using script defaults for filenames and output path. "
                   "Ensure PYTHONPATH is set correctly or run from project root if 'hybrid_search_rag' is a module there.")
    # Define fallback constants if config is not available,
    # aligning with the user's provided config.py values for consistency.
    class FallbackConfig:
        PROJECT_ROOT = os.getcwd() # Fallback project root
        # Default filenames from user's config.py
        METADATA_FILE = "combined_metadata.json"
        EMBEDDINGS_FILE = "combined_embeddings.npy" # Aligned
        BM25_INDEX_FILE = "bm25_index.pkl"
        # DATA_DIR is not used by this script for output, but defined for completeness if needed elsewhere
        DATA_DIR = os.path.join(PROJECT_ROOT, "data_hybrid_fallback")


    config = FallbackConfig() # Use the fallback class instance as 'config'
    logger.info(f"Using fallback PROJECT_ROOT: {config.PROJECT_ROOT}")
    logger.info(f"Using fallback METADATA_FILE: {config.METADATA_FILE}")
    logger.info(f"Using fallback EMBEDDINGS_FILE: {config.EMBEDDINGS_FILE}")
    logger.info(f"Using fallback BM25_INDEX_FILE: {config.BM25_INDEX_FILE}")


class DataManager:
    """Manages loading and saving of resource metadata, embeddings, and BM25 index."""
    def __init__(self, data_dir: str, metadata_filename: str, embeddings_filename: str, bm25_filename: str):
        """Initializes paths and ensures the data directory exists."""
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, metadata_filename)
        self.embeddings_path = os.path.join(data_dir, embeddings_filename)
        self.bm25_path = os.path.join(data_dir, bm25_filename)
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create data directory {self.data_dir}: {e}")
            raise

    def save_all_data(self, metadata: List[Dict[str, Any]], embeddings: Optional[np.ndarray], bm25_index: Optional[object]):
        """Saves metadata, embeddings (NumPy), and BM25 index (pickle).
        If metadata_path ends with .jsonl, metadata is saved in JSON Lines format.
        Otherwise, it's saved as a standard JSON array.
        """
        if not metadata:
            logger.warning("Attempted to save empty metadata list. Skipping save.")
            return

        if embeddings is not None and len(metadata) != embeddings.shape[0]:
            logger.error(f"Metadata count ({len(metadata)}) mismatch with embeddings count ({embeddings.shape[0]})! Aborting save.")
            raise ValueError("Metadata and embeddings counts must match.")

        try:
            # Save metadata
            logger.info(f"Saving metadata ({len(metadata)} items) to {self.metadata_path}")
            is_jsonl = self.metadata_path.lower().strip().endswith('.jsonl')
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                if is_jsonl:
                    logger.info(f"Saving metadata in JSONL format to {self.metadata_path}")
                    for item in metadata:
                        f.write(json.dumps(item, ensure_ascii=False) + '\\n')
                else:
                    logger.info(f"Saving metadata in standard JSON format to {self.metadata_path}")
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Save embeddings
            if embeddings is not None:
                logger.info(f"Saving embeddings (shape: {embeddings.shape}) to {self.embeddings_path}")
                np.save(self.embeddings_path, embeddings, allow_pickle=False)
            else:
                logger.info(f"No embeddings data provided, skipping save to {self.embeddings_path}")
                if os.path.exists(self.embeddings_path):
                    try:
                        os.remove(self.embeddings_path)
                        logger.info(f"Removed existing embeddings file: {self.embeddings_path}")
                    except OSError as e:
                         logger.warning(f"Could not remove existing embeddings file {self.embeddings_path}: {e}")

            # Save BM25 index
            if bm25_index is not None:
                logger.info(f"Saving BM25 index to {self.bm25_path}")
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump(bm25_index, f)
            else:
                logger.info(f"No BM25 index provided, skipping save to {self.bm25_path}")
                if os.path.exists(self.bm25_path):
                     try:
                         os.remove(self.bm25_path)
                         logger.info(f"Removed existing BM25 index file: {self.bm25_path}")
                     except OSError as e:
                          logger.warning(f"Could not remove existing BM25 index file {self.bm25_path}: {e}")

            logger.info(f"Data saving process completed successfully for directory: {self.data_dir}") # Corrected log
        except IOError as e:
            logger.error(f"IOError during file writing in {self.data_dir}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data saving in {self.data_dir}: {e}", exc_info=True)
            raise

    def load_all_data(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray], Optional[object]]:
        """Loads metadata, embeddings, and BM25 index from disk."""
        metadata: Optional[List[Dict[str, Any]]] = None
        embeddings: Optional[np.ndarray] = None
        bm25_index: Optional[object] = None
        logger.info(f"Attempting to load data from directory: {self.data_dir}")

        if not os.path.exists(self.metadata_path):
            logger.warning(f"Metadata file missing: {self.metadata_path}. Cannot load data.")
            return None, None, None
        try:
            logger.info(f"[DEBUG] Checking metadata_path: \'{self.metadata_path}\'") # Added for debugging
            is_jsonl = self.metadata_path.lower().strip().endswith('.jsonl') # Added for debugging
            logger.info(f"[DEBUG] Path ends with .jsonl: {is_jsonl}") # Added for debugging

            logger.info(f"Loading metadata from {self.metadata_path} (format: {'JSONL' if is_jsonl else 'JSON'})") # Modified log
            metadata_content = []
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                if is_jsonl: # Use the debugged variable
                    for line_number, line in enumerate(f, 1):
                        try:
                            # Skip empty or whitespace-only lines
                            if not line.strip():
                                continue
                            metadata_content.append(json.loads(line))
                        except json.JSONDecodeError as e_line: # Renamed to avoid clash with outer 'e'
                            logger.error(f"Error decoding JSON on line {line_number} in {self.metadata_path}: {e_line}")
                            continue 
                else: # Assume it's a standard JSON file
                    metadata_content = json.load(f)
            
            if not isinstance(metadata_content, list):
                 logger.error(f"Metadata file format error in {self.metadata_path}: Expected a JSON list.")
                 return None, None, None
            metadata = metadata_content
            logger.info(f"Loaded {len(metadata)} metadata items from {self.metadata_path}.")
        except (json.JSONDecodeError, IOError) as e:
             logger.error(f"Failed to load or decode metadata file {self.metadata_path}: {e}")
             return None, None, None
        except Exception as e:
             logger.error(f"Unexpected error loading metadata from {self.metadata_path}: {e}", exc_info=True)
             return None, None, None

        if not os.path.exists(self.embeddings_path):
            logger.warning(f"Embeddings file not found: {self.embeddings_path}. No embeddings will be loaded from this source.")
        else:
            try:
                logger.info(f"Loading embeddings from {self.embeddings_path}")
                embeddings = np.load(self.embeddings_path, allow_pickle=False)
                if metadata and embeddings is not None and len(metadata) != embeddings.shape[0]:
                    logger.error(f"Data mismatch in {self.data_dir}: Metadata count ({len(metadata)}) differs from loaded embeddings count ({embeddings.shape[0]}). Discarding embeddings from this source.")
                    embeddings = None
                elif metadata and embeddings is not None:
                    logger.info(f"Loaded embeddings with shape: {embeddings.shape} from {self.embeddings_path}.")
                elif not metadata and embeddings is not None:
                     logger.error(f"Loaded embeddings from {self.embeddings_path} but metadata is invalid?! Discarding embeddings.")
                     embeddings = None
            except (IOError, ValueError) as e:
                logger.error(f"Failed to load embeddings file {self.embeddings_path} (corrupt?): {e}")
                embeddings = None
            except Exception as e:
                 logger.error(f"Unexpected error loading embeddings from {self.embeddings_path}: {e}", exc_info=True)
                 embeddings = None

        if not os.path.exists(self.bm25_path):
             logger.warning(f"BM25 index file not found: {self.bm25_path}. No BM25 index will be loaded from this source.")
        else:
            try:
                logger.info(f"Loading BM25 index from {self.bm25_path}")
                with open(self.bm25_path, 'rb') as f:
                    bm25_index = pickle.load(f)
                logger.info(f"Loaded BM25 index from {self.bm25_path}.")
            except (IOError, pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError) as e:
                logger.error(f"Failed to load or unpickle BM25 index from {self.bm25_path}: {e}")
                bm25_index = None
            except Exception as e:
                logger.error(f"Unexpected error loading BM25 index from {self.bm25_path}: {e}", exc_info=True)
                bm25_index = None

        logger.info(f"Data loading attempt finished for directory: {self.data_dir}")
        return metadata, embeddings, bm25_index

# --- Main Script Logic ---
def combine_and_deduplicate_datasets(
    source_data_dirs: List[str],
    output_data_dir: str,
    # Default filenames are now taken from the imported (or fallback) config object
    metadata_filename: str = config.METADATA_FILE,
    embeddings_filename: str = config.EMBEDDINGS_FILE,
    bm25_filename: str = config.BM25_INDEX_FILE,
    # These keys are data-specific and not typically in a general config file
    unique_key_for_metadata: str = "id",
    text_content_key_for_bm25: str = "text"
):
    logger.info("Starting dataset combination and de-duplication process...")
    all_loaded_metadata: List[Dict[str, Any]] = []
    all_loaded_embeddings_rows: List[Optional[np.ndarray]] = [] # Allow None for missing embeddings

    logger.info("--- Step 1: Loading data from source directories ---")
    for i, data_path in enumerate(source_data_dirs): # Renamed data_dir_path to data_path for clarity
        logger.info(f"Processing source {i+1}: {data_path}")
        
        current_data_dir = data_path
        current_metadata_filename = metadata_filename # Default from function args
        current_embeddings_filename = embeddings_filename # Default from function args
        current_bm25_filename = bm25_filename # Default from function args

        if os.path.isfile(data_path):
            logger.info(f"Source {data_path} is a file. Assuming it's a metadata file.")
            current_data_dir = os.path.dirname(data_path)
            current_metadata_filename = os.path.basename(data_path)
            
            base_name, _ = os.path.splitext(current_metadata_filename)
            inferred_embeddings_file = base_name + ".npy"
            inferred_bm25_file = base_name + ".pkl"

            potential_embeddings_path = os.path.join(current_data_dir, inferred_embeddings_file)
            if os.path.exists(potential_embeddings_path):
                current_embeddings_filename = inferred_embeddings_file
                logger.info(f"Using inferred embeddings file for {data_path}: {potential_embeddings_path}")
            else:
                logger.info(f"Inferred embeddings file {potential_embeddings_path} not found for {data_path}. Will use default/passed: {current_embeddings_filename}")

            potential_bm25_path = os.path.join(current_data_dir, inferred_bm25_file)
            if os.path.exists(potential_bm25_path):
                current_bm25_filename = inferred_bm25_file
                logger.info(f"Using inferred BM25 file for {data_path}: {potential_bm25_path}")
            else:
                logger.info(f"Inferred BM25 file {potential_bm25_path} not found for {data_path}. Will use default/passed: {current_bm25_filename}")
        
        elif not os.path.isdir(data_path):
            logger.warning(f"Source path {data_path} is not a valid file or directory. Skipping.")
            continue
        
        # logger.info(f"Loading data from source {i+1}: {data_dir_path}") # Original log line, can be removed or adapted
        # Use filenames passed to this function (which default to config values, or are now inferred)
        dm = DataManager(current_data_dir, current_metadata_filename, current_embeddings_filename, current_bm25_filename)
        metadata, embeddings, _ = dm.load_all_data()

        if metadata:
            for k, meta_item in enumerate(metadata):
                all_loaded_metadata.append(meta_item)
                if embeddings is not None and k < embeddings.shape[0]:
                    all_loaded_embeddings_rows.append(embeddings[k])
                else:
                    all_loaded_embeddings_rows.append(None)
            logger.info(f"Successfully processed {len(metadata)} items from {data_path}.")
        else:
            logger.warning(f"No metadata loaded from {data_path}. Skipping this source.")

    if not all_loaded_metadata:
        logger.error("No metadata loaded from any source. Aborting.")
        return

    logger.info(f"Total metadata items loaded before de-duplication: {len(all_loaded_metadata)}")
    logger.info(f"Total embedding rows loaded (incl. None): {len(all_loaded_embeddings_rows)}")

    logger.info("--- Step 2: De-duplicating metadata and aligning embeddings ---")
    final_unique_metadata: List[Dict[str, Any]] = []
    # Store (unique_meta_item, corresponding_embedding_row_or_None)
    unique_meta_embed_pairs: List[Tuple[Dict[str, Any], Optional[np.ndarray]]] = []
    seen_identifiers = set()

    for i, meta_item in enumerate(all_loaded_metadata):
        identifier = meta_item.get(unique_key_for_metadata)
        if identifier is None:
            identifier = f"NO_ID_{str(meta_item)}"
            logger.warning(f"Metadata item at original index {i} has no '{unique_key_for_metadata}' key. Using its string representation as identifier: '{identifier[:50]}...'")

        if identifier not in seen_identifiers:
            seen_identifiers.add(identifier)
            unique_meta_embed_pairs.append((meta_item, all_loaded_embeddings_rows[i]))

    logger.info(f"Total unique metadata items after de-duplication: {len(unique_meta_embed_pairs)}")

    # Prepare final metadata and embeddings for saving
    final_metadata_for_save: List[Dict[str, Any]] = []
    final_embeddings_list_for_stacking: List[np.ndarray] = [] # Only non-None embeddings

    # Decide on saving strategy for embeddings
    # If any unique item has an embedding, we'll save embeddings.
    # In that case, metadata list will be filtered to only those with embeddings.
    # Otherwise, all unique metadata is saved, and embeddings are None.
    
    has_at_least_one_embedding_for_unique = any(embed is not None for _, embed in unique_meta_embed_pairs)

    if has_at_least_one_embedding_for_unique:
        logger.info("Embeddings found for some unique items. Final metadata will be filtered to items with embeddings.")
        for meta, embed in unique_meta_embed_pairs:
            if embed is not None:
                final_metadata_for_save.append(meta)
                final_embeddings_list_for_stacking.append(embed)
        final_embeddings_array = np.vstack(final_embeddings_list_for_stacking) if final_embeddings_list_for_stacking else None
        if final_embeddings_array is not None:
             logger.info(f"Final embeddings array shape: {final_embeddings_array.shape} for {len(final_metadata_for_save)} metadata items.")
        else: # Should not happen if has_at_least_one_embedding_for_unique is true and list is not empty
             logger.warning("Logic error: has_at_least_one_embedding_for_unique was true, but no embeddings to stack.")
             final_metadata_for_save = [meta for meta, _ in unique_meta_embed_pairs] # Save all unique meta
             final_embeddings_array = None

    else:
        logger.info("No embeddings found for any unique items, or all were None. Saving all unique metadata without an embeddings file.")
        final_metadata_for_save = [meta for meta, _ in unique_meta_embed_pairs]
        final_embeddings_array = None
    
    if not final_metadata_for_save:
        logger.error("No metadata to save after processing. Aborting save.")
        return

    logger.info("--- Step 3: Rebuilding BM25 index ---")
    new_bm25_index: Optional[object] = None
    corpus_for_bm25_rebuild = []
    for item in final_metadata_for_save: # Use the metadata list that will actually be saved
        text_content = item.get(text_content_key_for_bm25)
        if text_content:
            corpus_for_bm25_rebuild.append(text_content)
        else:
            logger.warning(f"Item with ID '{item.get(unique_key_for_metadata)}' in the final save list is missing '{text_content_key_for_bm25}' for BM25.")

    if corpus_for_bm25_rebuild:
        logger.info(f"Rebuilding BM25 index with {len(corpus_for_bm25_rebuild)} documents.")
        # IMPORTANT: Replace this with your actual BM25 library and initialization.
        # Example using a hypothetical 'rank_bm25' library (ensure it's installed: pip install rank-bm25):
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.split(" ") for doc in corpus_for_bm25_rebuild] # Example tokenization
            new_bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index rebuilt successfully using BM25Okapi.")
        except ImportError:
            logger.warning("`rank-bm25` library not found. Cannot rebuild BM25 index. Please install it (`pip install rank-bm25`).")
            new_bm25_index = None
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index with BM25Okapi: {e}", exc_info=True)
            new_bm25_index = None
    else:
        logger.warning("No text content found for rebuilding BM25 index. BM25 index will be None.")

    logger.info("--- Step 4: Saving combined and de-duplicated data ---")
    # output_data_dir is passed as a parameter
    # Filenames for saving will be based on the (config or fallback) defaults, prefixed with "final_"
    output_dm = DataManager(
        data_dir=output_data_dir,
        metadata_filename=f"final_{metadata_filename.replace('.json', '.jsonl') if not metadata_filename.endswith('.jsonl') else metadata_filename}", # Ensure output is .jsonl
        embeddings_filename=f"final_{embeddings_filename}",
        bm25_filename=f"final_{bm25_filename}"
    )
    
    try:
        output_dm.save_all_data(
            metadata=final_metadata_for_save,
            embeddings=final_embeddings_array,
            bm25_index=new_bm25_index
        )
        logger.info(f"Successfully saved combined data to {output_data_dir}")
    except Exception as e:
        logger.error(f"Failed to save combined data: {e}", exc_info=True)

    logger.info("Dataset combination and de-duplication process finished.")


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Specify the paths to your source dataset directories
    source_dirs = [
        # Example:
        os.path.join(config.PROJECT_ROOT, "arxiv_dataset/arxiv_papers_collected_feed.jsonl"),
        os.path.join(config.PROJECT_ROOT, "arxiv_dataset/final_combined_metadata.json"),
    ]
    
    # Define output directory using config.PROJECT_ROOT.
    # This is where the *output of this script* will go.
    # It's distinct from config.DATA_DIR which is for the main application's input.
    output_dir = os.path.join(config.PROJECT_ROOT, "arxiv_dataset/data_combined_script_output")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    logger.info(f"Combined data will be saved to: {output_dir}")


    # Create dummy data for testing if no source_dirs are provided
    if not source_dirs:
        logger.warning("No source_dirs provided. Creating dummy data for demonstration.")
        dummy_base_path = os.path.join(config.PROJECT_ROOT, "dummy_data_sources_for_script")
        os.makedirs(dummy_base_path, exist_ok=True)
        
        for i in range(1, 3): # Create 2 dummy sources
            s_dir = os.path.join(dummy_base_path, f"source_{i}")
            os.makedirs(s_dir, exist_ok=True)
            source_dirs.append(s_dir)
            
            dummy_meta = []
            for j in range(3):
                item_id_val = f"item_id_{i}_{j}"
                if j == 0: item_id_val = "common_id_0" # Duplicate across sources
                if j == 2 and i == 1 : item_id_val = "common_id_0" 
                elif j == 2 and i == 2 : item_id_val = "another_common_id_2"

                dummy_meta.append({
                    "id": item_id_val, 
                    "text": f"This is text for item {j} from source {i}. ID: {item_id_val}",
                    "source_file": f"file_{i}_{j}.txt"
                })
            # Use config filenames for dummy data
            with open(os.path.join(s_dir, config.METADATA_FILE), "w") as f:
                json.dump(dummy_meta, f, indent=2)

            if i == 1: # Only source 1 has embeddings for this demo
                dummy_embeds = np.random.rand(len(dummy_meta), 10)
                np.save(os.path.join(s_dir, config.EMBEDDINGS_FILE), dummy_embeds)
        logger.info(f"Created dummy source directories for testing: {source_dirs}")

    # IMPORTANT: Specify the key in your metadata for unique identification
    unique_id_key = "id" 
    # IMPORTANT: Specify the key in your metadata for text content for BM25
    bm25_text_key = "text"

    if not source_dirs:
        logger.error("Source directories are not specified and dummy data creation might have issues if config.PROJECT_ROOT is not as expected. Please update the 'source_dirs' list or check paths.")
    else:
        # Call the main function.
        # It will use defaults from the imported/fallback config for filenames.
        combine_and_deduplicate_datasets(
            source_data_dirs=source_dirs,
            output_data_dir=output_dir,
            # metadata_filename, embeddings_filename, bm25_filename will use defaults from config
            unique_key_for_metadata=unique_id_key,
            text_content_key_for_bm25=bm25_text_key
        )
