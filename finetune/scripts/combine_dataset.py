# combine_datasets.py

import json
import os
import sys
import logging
# from tqdm import tqdm # Not directly used by the centralized function in console output

# --- Add project root to sys.path for module imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import the main function and config 
    from hybrid_search_rag.data_handling import combine_and_deduplicate_datasets
    from hybrid_search_rag import config as dm_config
except ImportError as e:
    print(f"Error importing from hybrid_search_rag.data_manager: {e}", file=sys.stderr)
    print("Please ensure that the project root is correctly added to PYTHONPATH or sys.path,", file=sys.stderr)
    print(f"and the hybrid_search_rag package is available from: {project_root}", file=sys.stderr)
    sys.exit(1)
# --- End sys.path modification and import ---


# --- Configuration ---

# Define basenames for input files within the dataset directory
INPUT_FILE_BASENAMES = [
    "arxiv_cs_papers_dataset_selective.jsonl",
    "arxiv_cs_papers_dataset.jsonl"
]
# Define basename for the output file within the dataset directory
# The combine_and_deduplicate_datasets function will prefix this with "final_"
# and ensure it has a .jsonl extension.
COMBINED_OUTPUT_BASENAME = "arxiv_cs_papers_dataset_final.jsonl" 

# Field to use for deduplication (passed to the centralized function)
ID_FIELD = 'id'
# Field for text content for BM25 (passed to the centralized function)
TEXT_CONTENT_KEY = 'text' # Assuming 'text' is the field in your JSONL for BM25

# --- Setup Logger ---
log_dir = os.path.join(os.path.dirname(__file__), "logs") # Place logs in scripts/logs
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "combine_datasets_script.log") # Renamed log for clarity

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a') # Append mode
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("CombineDatasetScript") # Renamed logger for clarity
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False
# --- End Logger Setup ---

def main():
    """Main function to run the dataset combination process."""
    print("--- Starting Dataset Combination Script (using centralized function) ---")
    logger.info("Starting dataset combination using centralized combine_and_deduplicate_datasets function.")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Path to the 'finetune/dataset/' directory, which will serve as the base for inputs and outputs
    # This path is where the input files are located and where the output directory will be created.
    dataset_base_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))

    # The combine_and_deduplicate_datasets function expects full paths for source_data_dirs if they are files,
    # or directory paths. Here, we construct full paths to the input files.
    source_full_paths = []
    for basename in INPUT_FILE_BASENAMES:
        full_path = os.path.join(dataset_base_dir, basename)
        if not os.path.exists(full_path):
            logger.error(f"Input file not found: {full_path}")
            print(f"ERROR: Input file not found: {full_path}", file=sys.stderr)
            sys.exit(1)
        source_full_paths.append(full_path)
    
    # The output_data_dir is where the final combined files will be stored.
    # Let's make it a subdirectory within dataset_base_dir for clarity, e.g., 'dataset/combined_output/'
    # The combine_and_deduplicate_datasets function will create this directory if it doesn't exist.
    output_directory_for_combined_data = os.path.join(dataset_base_dir, "combined_output_from_script")
    # No need to create it here, combine_and_deduplicate_datasets handles it via DataManager

    logger.info(f"Source files to combine: {source_full_paths}")
    logger.info(f"Output directory for combined data: {output_directory_for_combined_data}")
    logger.info(f"Output metadata basename (will be prefixed 'final_'): {COMBINED_OUTPUT_BASENAME}")
    logger.info(f"ID field for deduplication: {ID_FIELD}")
    logger.info(f"Text content key for BM25: {TEXT_CONTENT_KEY}")

    try:
        # Call the centralized function
        # It uses filenames from its own config (or fallback) for the components *within* the output_data_dir
        # We provide the base name for the output metadata file.
        combine_and_deduplicate_datasets(
            source_data_dirs=source_full_paths, # List of full paths to source files/dirs
            output_data_dir=output_directory_for_combined_data, # Directory where 'final_...' files will be saved
            metadata_filename=COMBINED_OUTPUT_BASENAME, # Base name for the output metadata file
            # embeddings_filename and bm25_filename will use defaults from dm_config
            # or be inferred if source_data_dirs are files with associated .npy/.pkl files.
            unique_key_for_metadata=ID_FIELD,
            text_content_key_for_bm25=TEXT_CONTENT_KEY
        )
        
        final_output_file_path = os.path.join(output_directory_for_combined_data, f"final_{COMBINED_OUTPUT_BASENAME}")
        logger.info(f"Combination complete. Main unique dataset expected at: {final_output_file_path}")
        print(f"\nCombination complete. Main unique dataset expected at: {final_output_file_path}")
        print(f"Check the log file for details: {log_file_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found during combination: {e}", exc_info=True)
        print(f"\nError: A required file was not found: {e}", file=sys.stderr)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the combination process: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        print(f"Check the log file for details: {log_file_path}")

    print("--- Script Finished ---")

if __name__ == "__main__":
    main()