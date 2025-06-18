# -*- coding: utf-8 -*-
"""
Dataset Combiner for StudyAssistant Project

Provides functionality to combine and deduplicate datasets from multiple sources.
"""

import os
import json
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

from .data_manager import DataManager

logger = logging.getLogger(__name__)

def combine_and_deduplicate_datasets(
    source_data_dirs: List[str],
    output_data_dir: str,
    metadata_filename: str = "combined_metadata.json",
    embeddings_filename: str = "combined_embeddings.npy",
    bm25_filename: str = "bm25_index.pkl",
    unique_key_for_metadata: str = "id",
    text_content_key_for_bm25: str = "text"
):
    """
    Combines and deduplicates datasets from multiple source directories.
    
    Args:
        source_data_dirs: List of paths to source data directories or files
        output_data_dir: Directory to save combined output
        metadata_filename: Name of metadata file
        embeddings_filename: Name of embeddings file
        bm25_filename: Name of BM25 index file
        unique_key_for_metadata: Key to use for deduplication
        text_content_key_for_bm25: Key containing text content for BM25
    """
    logger.info("Starting dataset combination and de-duplication process...")
    all_loaded_metadata: List[Dict[str, Any]] = []
    all_loaded_embeddings_rows: List[Optional[np.ndarray]] = []

    logger.info("--- Step 1: Loading data from source directories ---")
    for i, data_path in enumerate(source_data_dirs):
        logger.info(f"Processing source {i+1}: {data_path}")
        
        current_data_dir = data_path
        current_metadata_filename = metadata_filename
        current_embeddings_filename = embeddings_filename
        current_bm25_filename = bm25_filename

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
                logger.info(f"Inferred embeddings file {potential_embeddings_path} not found for {data_path}. Will use default: {current_embeddings_filename}")

            potential_bm25_path = os.path.join(current_data_dir, inferred_bm25_file)
            if os.path.exists(potential_bm25_path):
                current_bm25_filename = inferred_bm25_file
                logger.info(f"Using inferred BM25 file for {data_path}: {potential_bm25_path}")
            else:
                logger.info(f"Inferred BM25 file {potential_bm25_path} not found for {data_path}. Will use default: {current_bm25_filename}")
        
        elif not os.path.isdir(data_path):
            logger.warning(f"Source path {data_path} is not a valid file or directory. Skipping.")
            continue
        
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

    final_metadata_for_save: List[Dict[str, Any]] = []
    final_embeddings_list_for_stacking: List[np.ndarray] = []

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
        else:
             logger.warning("Logic error: has_at_least_one_embedding_for_unique was true, but no embeddings to stack.")
             final_metadata_for_save = [meta for meta, _ in unique_meta_embed_pairs]
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
    for item in final_metadata_for_save:
        text_content = item.get(text_content_key_for_bm25)
        if text_content:
            corpus_for_bm25_rebuild.append(text_content)
        else:
            logger.warning(f"Item with ID '{item.get(unique_key_for_metadata)}' in the final save list is missing '{text_content_key_for_bm25}' for BM25.")

    if corpus_for_bm25_rebuild:
        logger.info(f"Rebuilding BM25 index with {len(corpus_for_bm25_rebuild)} documents.")
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.split(" ") for doc in corpus_for_bm25_rebuild]
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
    output_dm = DataManager(
        data_dir=output_data_dir,
        metadata_filename=f"final_{metadata_filename.replace('.json', '.jsonl') if not metadata_filename.endswith('.jsonl') else metadata_filename}",
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
