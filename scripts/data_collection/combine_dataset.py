# combine_datasets.py

import json
import os
import sys
import logging
from tqdm import tqdm

# --- Configuration (EDIT THESE PATHS IF NEEDED) ---
FILE1_PATH = "arxiv_dataset/arxiv_source_docs_kaggle_2.jsonl"
FILE2_PATH = "arxiv_dataset/arxiv_source_docs_kaggle.jsonl"
COMBINED_OUTPUT_PATH = "arxiv_dataset/arxiv_cs_papers_dataset_combine_1.jsonl"

# Field to use for deduplication
ID_FIELD = 'id'

# --- Setup Logger ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "combine_datasets.log")

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("CombineDatasets")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False
# --- End Logger Setup ---

def combine_and_deduplicate_jsonl(input_paths: list[str], output_path: str, id_field: str):
    """
    Reads multiple JSONL files, combines them, removes duplicates based on id_field,
    and writes the unique records to a new JSONL file.

    Args:
        input_paths: A list of paths to the input JSONL files.
        output_path: The path where the combined, unique JSONL file will be saved.
        id_field: The key in the JSON object to use for deduplication (e.g., 'id').
    """
    seen_ids = set()
    total_lines_read = 0
    duplicates_skipped = 0
    written_count = 0
    corrupt_lines = 0
    missing_id_lines = 0

    logger.info(f"Starting dataset combination and deduplication.")
    logger.info(f"Input files: {', '.join(input_paths)}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Deduplicating based on field: '{id_field}'")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for file_path in input_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"Input file not found, skipping: {file_path}")
                    continue

                logger.info(f"Processing file: {file_path}")
                try:
                    # Get file size for tqdm progress bar (optional but nice)
                    file_size = os.path.getsize(file_path)
                    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading {os.path.basename(file_path)}")
                except OSError:
                    progress_bar = None

                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            total_lines_read += 1
                            if progress_bar:
                                progress_bar.update(len(line.encode('utf-8')))

                            line_stripped = line.strip()
                            if not line_stripped:
                                continue

                            try:
                                data = json.loads(line_stripped)
                                paper_id = data.get(id_field)

                                if paper_id is None:
                                    logger.warning(f"Line {total_lines_read} in {file_path} is missing the '{id_field}' field. Skipping.")
                                    missing_id_lines += 1
                                    continue

                                if paper_id not in seen_ids:
                                    seen_ids.add(paper_id)
                                    outfile.write(line)
                                    written_count += 1
                                else:
                                    duplicates_skipped += 1
                                    if duplicates_skipped % 1000 == 0:
                                         logger.debug(f"Skipped {duplicates_skipped} duplicate records so far...")


                            except json.JSONDecodeError:
                                logger.warning(f"Line {total_lines_read} in {file_path} is corrupt or not valid JSON. Skipping.")
                                corrupt_lines += 1
                            except Exception as e:
                                logger.error(f"Unexpected error processing line {total_lines_read} in {file_path}: {e}", exc_info=True)
                                corrupt_lines += 1

                except Exception as e:
                     logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
                finally:
                     if progress_bar:
                          progress_bar.close()


        logger.info("--- Combination Summary ---")
        logger.info(f"Total lines read across all files: {total_lines_read}")
        logger.info(f"Unique records written to '{output_path}': {written_count}")
        logger.info(f"Duplicate records skipped: {duplicates_skipped}")
        logger.info(f"Lines skipped due to missing '{id_field}': {missing_id_lines}")
        logger.info(f"Lines skipped due to JSON errors/corruption: {corrupt_lines}")
        logger.info("--------------------------")

    except IOError as e:
        logger.error(f"Failed to open or write file. Check permissions and path '{output_path}'. Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the process: {e}", exc_info=True)

if __name__ == "__main__":
    print("--- Starting Dataset Combination Script ---")
    input_files = [FILE1_PATH, FILE2_PATH]

    files_exist = True
    for f_path in input_files:
        if not os.path.exists(f_path):
            print(f"ERROR: Input file not found: {f_path}", file=sys.stderr)
            logger.error(f"Prerequisite check failed: Input file not found: {f_path}")
            files_exist = False

    if files_exist:
        combine_and_deduplicate_jsonl(
            input_paths=input_files,
            output_path=COMBINED_OUTPUT_PATH,
            id_field=ID_FIELD
        )
        print(f"\nCombination complete. Unique dataset saved to: {COMBINED_OUTPUT_PATH}")
        print(f"Check the log file for details: {log_file_path}")
    else:
        print("\nAborting script because one or more input files were not found.", file=sys.stderr)
        print("Please ensure the file paths at the top of the script are correct.", file=sys.stderr)

    print("--- Script Finished ---")