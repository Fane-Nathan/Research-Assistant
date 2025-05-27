# arxiv_local_processor_parallel.py
# Parallelized local version of the arXiv data collection script
# MODIFIED: Uses ThreadPoolExecutor for parallelism
# MODIFIED: Only outputs selected metadata fields + fetched data

import os
import sys
import logging
import time
import requests
import fitz
import json
import io
import re
import random
from tqdm import tqdm
import arxiv
import tenacity
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Dict, Any, Tuple, Optional

# --- Parameters ---

# <<< Number of parallel worker threads >>>
NUM_WORKERS = 12

# 1. Input file:
KAGGLE_METADATA_FILE = "/home/felixnathaniel/Documents/MachineLearning/StudyAssistant/finetune/dataset/arxiv-metadata-oai-snapshot.json" # !!! EDIT THIS PATH !!!

# 2. Final output file name (will be created/appended in the specified path)
FINAL_OUTPUT_FILE = "/home/felixnathaniel/Documents/MachineLearning/StudyAssistant/finetune/dataset/arxiv_cs_papers_dataset_selective_parallel.jsonl" # !!! Using a new name !!!

# 3. List of categories to INCLUDE (paper must have AT LEAST ONE matching prefix)
#    MODIFIED: Includes ALL standard cs.* AND math.* categories + stat.ML
FILTER_CATEGORIES = sorted(list(set([
    # --- Computer Science Categories ---
    'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV',
    'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL',
    'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
    'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS',
    'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY',
    # --- Mathematics Categories ---
    'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA', 'math.CO',
    'math.CT', 'math.CV', 'math.DG', 'math.DS', 'math.FA', 'math.GM',
    'math.GN', 'math.GR', 'math.GT', 'math.HO', 'math.IT', 'math.KT',
    'math.LO', 'math.MG', 'math.MP', 'math.NA', 'math.NT', 'math.OA',
    'math.OC', 'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG',
    'math.SP', 'math.ST',
    # --- Relevant Cross-listed / Other ---
    'stat.ML'
]))) # Use set to ensure uniqueness (e.g., for cs.IT/math.IT) and sort

# 4. Maximum number of *new* papers to SUBMIT for processing in THIS RUN
#    Note: More papers might be processed if script resumes. This limits new submissions.
#    Set to None to submit all matching papers from the metadata file.
MAX_PAPERS_TO_PROCESS = 100000 # Example: Target 100k *new* papers

# 5. Timeout in seconds for downloading each PDF
REQUESTS_TIMEOUT = 1

# 6. Delay in seconds BETWEEN STARTING network calls for EACH paper in a worker thread
ARXIV_API_DELAY_PER_ID = 0.5

# 7. Runtime Limit Configuration (Optional for Local Use)
MAX_RUNTIME_SECONDS = None # Set to None to disable time limit by default

# 8. Progress Logging Interval
LOG_INTERVAL_SECONDS = 300 # Log progress every 5 minutes
# --- End Parameters ---

# --- Setup Logger ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "data_collection_parallel.log")

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - [%(threadName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("DataCollectParallel")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False
# --- End Logger Setup ---

# --- Helper Functions ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]
def get_user_agent():
    return random.choice(USER_AGENTS)

def _clean_text(text: Union[str, None]) -> Union[str, None]:
    if not text: return None
    try:
        text = re.sub(r'[ \t]+', ' ', text)
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        if not cleaned_lines: return None
        text = "\n".join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        cleaned_text = text.strip()
        return cleaned_text if cleaned_text else None
    except Exception as e:
        logger.debug(f"Text cleaning failed: {e}")
        return text

def _parse_pdf_content(pdf_bytes: bytes, source_url: str) -> Union[str, None]:
    pdf_text = ""
    try:
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            if doc.is_encrypted and not doc.authenticate(''):
                 logger.warning(f"PDF is encrypted: {source_url}")
                 return None
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text("text", sort=True)
                    if page_text:
                        pdf_text += page_text + "\n\n"
                except Exception as page_err:
                    logger.warning(f"Error processing page {page_num+1} in PDF {source_url}: {page_err}")
                    continue
        cleaned_text = _clean_text(pdf_text)
        if not cleaned_text:
            logger.warning(f"No text extracted from PDF after cleaning: {source_url}")
        return cleaned_text
    except fitz.EmptyFileError:
         logger.warning(f"PDF processing failed: Empty/invalid PDF. {source_url}")
         return None
    except fitz.FileDataError as data_err:
         logger.warning(f"PDF processing failed: Data Error. {source_url}: {data_err}")
         return None
    except Exception as e:
        logger.warning(f"PDF processing failed: {e.__class__.__name__}. {source_url}: {e}")
        return None
# --- End Helper Functions ---

# --- Worker Function for Processing a Single Paper ---
def process_single_paper(metadata: Dict[str, Any], api_delay: float, fetch_timeout: int) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """
    Fetches details, downloads/parses PDF for a single paper's metadata.
    Designed to be run in a separate thread.

    Args:
        metadata: The metadata dictionary for the paper (must include 'id').
        api_delay: Delay in seconds before making network calls.
        fetch_timeout: Timeout for requests.

    Returns:
        A tuple: (paper_id, output_record | None, status_string)
        output_record is None if processing fails.
    """
    paper_id = metadata.get('id')
    if not paper_id:
        return ("UNKNOWN_ID", None, "Error: Missing ID in metadata")

    arxiv_client = arxiv.Client(page_size=1, delay_seconds=0.1, num_retries=3)

    full_text: Optional[str] = None
    pdf_status: str = "Not Attempted"
    pdf_url_from_api: Optional[str] = None
    output_record: Optional[Dict[str, Any]] = None

    try:
        actual_delay = api_delay + random.uniform(0, api_delay * 0.5)
        time.sleep(actual_delay)
        logger.debug(f"[{paper_id}] Starting processing after {actual_delay:.2f}s delay")

        search_by_id = arxiv.Search(id_list=[paper_id], max_results=1)
        found_result = None
        api_attempt_start = time.time()
        try:
            results_generator = arxiv_client.results(search_by_id)
            found_result = next(results_generator, None)
            api_attempt_duration = time.time() - api_attempt_start
            logger.debug(f"[{paper_id}] API lookup took {api_attempt_duration:.2f}s")
        except StopIteration:
            logger.warning(f"[{paper_id}] API returned no results for known ID.")
            pdf_status = "API Error: No results"
        except tenacity.RetryError as retry_err:
            logger.error(f"[{paper_id}] API call failed after retries: {retry_err}", exc_info=False)
            pdf_status = f"API Error: RetryError"
        except Exception as api_err:
            logger.error(f"[{paper_id}] API call failed: {api_err}", exc_info=False)
            pdf_status = f"API Error: {api_err.__class__.__name__}"

        # --- Download and Parse PDF ---
        if pdf_status == "Not Attempted":
            if found_result and found_result.pdf_url:
                pdf_url_from_api = found_result.pdf_url
                logger.debug(f"[{paper_id}] PDF URL: {pdf_url_from_api}. Downloading...")
                headers = {'User-Agent': get_user_agent()}
                try:
                    pdf_response = requests.get(pdf_url_from_api, timeout=fetch_timeout, headers=headers, stream=True)
                    pdf_response.raise_for_status()
                    content_type = pdf_response.headers.get("Content-Type", "").lower()

                    if 'application/pdf' in content_type:
                        try:
                            pdf_bytes = pdf_response.content
                            if len(pdf_bytes) == 0:
                                logger.warning(f"[{paper_id}] Downloaded PDF is empty.")
                                pdf_status = "Download Error: Empty File"
                            else:
                                logger.debug(f"[{paper_id}] Read {len(pdf_bytes)} bytes. Parsing...")
                                full_text = _parse_pdf_content(pdf_bytes, pdf_url_from_api)
                                if full_text:
                                    pdf_status = "Success"
                                    logger.debug(f"[{paper_id}] PDF parsed successfully.")
                                else:
                                    pdf_status = "Parsing Failed"
                        except requests.exceptions.RequestException as read_err:
                             logger.error(f"[{paper_id}] Error reading downloaded content: {read_err}. Using abstract.")
                             pdf_status = f"Download Error: Read Failure"
                        except Exception as parse_err:
                             logger.error(f"[{paper_id}] Unexpected error during PDF parsing: {parse_err}. Using abstract.", exc_info=True)
                             pdf_status = "Parsing Error: Unexpected"
                    else:
                        pdf_status = f"Wrong Content-Type: {content_type}"
                        logger.warning(f"[{paper_id}] Expected PDF, got {content_type}. Using abstract.")

                except requests.exceptions.Timeout:
                     logger.error(f"[{paper_id}] PDF download timed out ({fetch_timeout}s). Using abstract.")
                     pdf_status = f"Download Timeout"
                except requests.exceptions.RequestException as req_err:
                     http_status = getattr(req_err.response, 'status_code', 'N/A')
                     logger.error(f"[{paper_id}] PDF download error (HTTP {http_status}): {req_err}. Using abstract.")
                     pdf_status = f"Download Error: {req_err.__class__.__name__} (HTTP {http_status})"
                finally:
                    if 'pdf_response' in locals() and hasattr(pdf_response, 'close'):
                        pdf_response.close()
            else:
                pdf_status = "No PDF URL via API"
                if not found_result:
                    pdf_status = "Paper Not Found via API"
                logger.warning(f"[{paper_id}] Could not find PDF URL. Using abstract. Status: {pdf_status}")

        # --- Fallback to Abstract ---
        if pdf_status != "Success":
            full_text = metadata.get('abstract', 'Abstract not available.')
            if not full_text:
                 full_text = "Text not available."
                 logger.warning(f"[{paper_id}] Abstract also missing.")
                 if pdf_status == "Not Attempted": pdf_status = "No Text Available"

        # --- Create the selective output record ---
        output_record = {
            'id': paper_id,
            'title': metadata.get('title'),
            'authors': metadata.get('authors'),
            'categories': metadata.get('categories'),
            'abstract': metadata.get('abstract'),
            'doi': metadata.get('doi'),
            'full_text': full_text,
            'pdf_fetched_url': pdf_url_from_api,
            'pdf_status': pdf_status
        }
        return (paper_id, output_record, pdf_status)

    except Exception as e:
        logger.error(f"[{paper_id}] UNHANDLED error in worker thread: {e}", exc_info=True)
        # Try to return paper_id if possible, otherwise use a placeholder
        return (paper_id or "UNKNOWN_ERROR_ID", None, f"Worker Error: {e.__class__.__name__}")
# --- End Worker Function ---


# --- Main Orchestration Function ---
def run_parallel_processing(
    kaggle_metadata_file: str,
    output_file: str,
    filter_categories: list[str],
    max_papers_to_process_this_run: Optional[int],
    fetch_timeout: int,
    api_delay: float,
    max_runtime_seconds: Optional[int],
    log_interval_seconds: int,
    num_workers: int
):
    """ Orchestrates the parallel processing of arXiv metadata. """
    logger.info(f"--- Starting Parallel Data Processing Run ---")
    logger.info(f"Using {num_workers} worker threads.")
    logger.info(f"-------------------------------------------")

    papers_submitted_this_run = 0
    papers_considered = 0
    stats = {'processed_ok': 0, 'processed_fail': 0, 'already_processed_skipped': 0, 'category_mismatch_skipped': 0}
    total_processed_ids_in_file = 0

    # --- Checkpoint/Resume Logic ---
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            logger.info(f"Checking existing output file for resuming: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f_check:
                 try:
                     f_check.seek(0, os.SEEK_END)
                     size = f_check.tell()
                     f_check.seek(0)
                     line_count_est = size // 1000
                     id_load_iterator = tqdm(f_check, desc="Loading processed IDs", unit=" lines", total=line_count_est or None, mininterval=2.0)
                 except (io.UnsupportedOperation, OSError):
                     id_load_iterator = tqdm(f_check, desc="Loading processed IDs", unit=" lines", mininterval=2.0)

                 for line_num, line in enumerate(id_load_iterator):
                    try:
                        if line.strip():
                            data = json.loads(line)
                            if 'id' in data and data['id']:
                                processed_ids.add(data['id'])
                            else:
                                logger.warning(f"Resume check: Line {line_num+1} missing 'id' or empty ID in {output_file}")
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping corrupted JSON line {line_num+1} during resume check in {output_file}")
            total_processed_ids_in_file = len(processed_ids)
            logger.info(f"Resuming run. Found {total_processed_ids_in_file} already processed IDs.")
        except Exception as e:
            logger.error(f"Error reading existing output file '{output_file}' for resuming: {e}. Starting fresh.", exc_info=True)
            processed_ids = set()
    else:
        logger.info(f"Output file '{output_file}' not found. Starting fresh.")
    # --- End Checkpoint Logic ---

    start_run_time = time.time()
    last_log_time = start_run_time

    futures = [] # List to hold Future objects from submitted tasks

    try:
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='Worker') as executor, \
             open(kaggle_metadata_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'a', encoding='utf-8') as outfile:

            logger.info("Phase 1: Scanning metadata file and submitting tasks...")
            metadata_iterator = tqdm(infile, desc="Scanning Metadata", unit=" lines", mininterval=1.0, smoothing=0.1)

            # --- Phase 1: Scan metadata and submit tasks ---
            for line in metadata_iterator:
                papers_considered += 1

                current_submit_time = time.time()
                if max_runtime_seconds is not None and (current_submit_time - start_run_time) > max_runtime_seconds:
                    logger.warning("Reached runtime limit during submission phase. Stopping submission of new tasks.")
                    break

                # --- Limit number of *new* papers submitted ---
                if max_papers_to_process_this_run is not None and papers_submitted_this_run >= max_papers_to_process_this_run:
                    logger.info(f"Reached submission target of {max_papers_to_process_this_run} new papers for this run.")
                    break

                try:
                    if not line.strip(): continue
                    metadata = json.loads(line)
                except json.JSONDecodeError:
                    if papers_considered % 50000 == 0: logger.warning(f"Scanned ~{papers_considered}. JSON corruption. Skip.")
                    continue

                paper_id = metadata.get('id')
                if not paper_id:
                     if papers_considered % 50000 == 0: logger.warning(f"Metadata line ~{papers_considered} missing 'id'. Skip.")
                     continue

                # --- Apply Filters BEFORE Submission ---
                if paper_id in processed_ids:
                    stats['already_processed_skipped'] += 1
                    if papers_considered % 10000 == 0:
                         metadata_iterator.set_postfix_str(f"Skipped {stats['already_processed_skipped']} processed", refresh=False)
                    continue

                paper_categories_str = metadata.get('categories', '')
                category_match = False
                if paper_categories_str and isinstance(paper_categories_str, str):
                    paper_categories = paper_categories_str.split()
                    if any(p_cat.startswith(f_cat) for p_cat in paper_categories for f_cat in filter_categories):
                        category_match = True

                if not category_match:
                    stats['category_mismatch_skipped'] += 1
                    continue

                # --- Submit Task ---
                # Pass metadata (as dict) and config parameters needed by worker
                future = executor.submit(process_single_paper, metadata, api_delay, fetch_timeout)
                futures.append(future)
                papers_submitted_this_run += 1
                if papers_submitted_this_run % 100 == 0:
                     metadata_iterator.set_postfix_str(f"Submitted {papers_submitted_this_run}", refresh=True)


            logger.info(f"Phase 1 Complete. Submitted {papers_submitted_this_run} tasks for processing.")
            metadata_iterator.close()

            # --- Phase 2: Process results as they complete ---
            logger.info("Phase 2: Waiting for tasks to complete and writing results...")
            results_iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing Papers", unit=" paper", mininterval=1.0, smoothing=0.1)

            for future_num, future in enumerate(results_iterator):
                current_process_time = time.time()
                # Check runtime limit also during result processing
                if max_runtime_seconds is not None and (current_process_time - start_run_time) > max_runtime_seconds:
                     logger.warning("Reached runtime limit during result processing phase. Waiting for currently running tasks may continue briefly.")
                     # Optionally add logic to cancel pending futures if needed, but let's keep it simple
                     # executor.shutdown(wait=False, cancel_futures=True) # More complex handling
                     break # Stop processing results

                try:
                    paper_id, output_record, pdf_status = future.result()

                    if output_record:
                        outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                        outfile.flush()
                        stats['processed_ok'] += 1
                        processed_ids.add(paper_id)
                    else:
                        logger.warning(f"Processing failed for {paper_id}. Status: {pdf_status}")
                        stats['processed_fail'] += 1

                except Exception as e:
                    logger.error(f"Error retrieving result for a future: {e}", exc_info=True)
                    stats['processed_fail'] += 1

                # --- Periodic Progress Log ---
                if current_process_time - last_log_time > log_interval_seconds:
                    elapsed_total_time = current_process_time - start_run_time
                    processed_total = stats['processed_ok'] + stats['processed_fail']
                    papers_per_sec = processed_total / elapsed_total_time if elapsed_total_time > 0 else 0
                    remaining_futures = len(futures) - (future_num + 1)
                    eta_str = f"~{(remaining_futures / papers_per_sec) / 3600:.1f}h" if papers_per_sec > 0 and remaining_futures > 0 else "N/A"

                    if max_runtime_seconds is not None:
                         remaining_runtime_sec = max(0, max_runtime_seconds - elapsed_total_time)
                         if papers_per_sec <= 0 or (remaining_futures / papers_per_sec > remaining_runtime_sec):
                              eta_str = f"~{remaining_runtime_sec / 3600:.1f}h (Runtime Limit)"

                    progress_runtime_str = f"{elapsed_total_time / 3600:.2f}/{max_runtime_seconds / 3600:.1f}h" if max_runtime_seconds is not None else f"{elapsed_total_time / 3600:.2f}h"

                    logger.info(
                         f"Progress: {stats['processed_ok'] + stats['processed_fail']}/{len(futures)} submitted | "
                         f"Total Saved: {total_processed_ids_in_file + stats['processed_ok']} | "
                         f"Elapsed: {progress_runtime_str} | "
                         f"Rate: {papers_per_sec:.2f} papers/s | "
                         f"OK/Fail: {stats['processed_ok']}/{stats['processed_fail']} | "
                         f"ETA: {eta_str}"
                    )
                    results_iterator.set_postfix_str(f"OK:{stats['processed_ok']} Fail:{stats['processed_fail']}", refresh=True)
                    last_log_time = current_process_time

            results_iterator.close()
            logger.info("Phase 2 Complete. Finished processing results.")

    except FileNotFoundError as fnf_err:
         logger.error(f"Input file not found: {fnf_err}")
         print(f"Error: Input file '{kaggle_metadata_file}' not found.", file=sys.stderr)
    except IOError as io_err:
        logger.error(f"File I/O error: {io_err}", exc_info=True)
        print(f"Error accessing files: {io_err}. Check permissions and paths.", file=sys.stderr)
    except KeyboardInterrupt:
         logger.warning("KeyboardInterrupt received. Shutting down executor (may take time)...")
         # Executor shutdown is handled by the 'with' statement exiting
         print("\nProcessing interrupted by user. Results processed so far are saved.")
    except Exception as e:
         logger.error(f"An unexpected error occurred outside the main processing loop: {e}", exc_info=True)
         print(f"An unexpected error occurred: {e}", file=sys.stderr)

    # --- Final Summary Logging ---
    logger.info(f"--- Final Run Summary ---")
    logger.info(f"Metadata Records Scanned: {papers_considered}")
    logger.info(f"Papers Submitted This Run: {papers_submitted_this_run}")
    logger.info(f"Papers Processed Successfully: {stats['processed_ok']}")
    logger.info(f"Papers Failed Processing: {stats['processed_fail']}")
    logger.info(f"Total Unique Papers Now in Output File ('{output_file}'): {total_processed_ids_in_file + stats['processed_ok']}")
    logger.info(f"Skipped (Already Processed): {stats['already_processed_skipped']}")
    logger.info(f"Skipped (Category Mismatch): {stats['category_mismatch_skipped']}")
    total_runtime = time.time() - start_run_time
    logger.info(f"Total Runtime This Session: {total_runtime:.2f} seconds ({total_runtime/3600:.2f} hours)")
    logger.info(f"Log file saved to: {log_file_path}")
    logger.info(f"---------------------------")
    return stats['processed_ok']


# --- Verification Function (Optional Call) ---
def verify_output_file(output_file_path: str, lines_to_show: int = 5):
    """Reads the beginning of the output JSONL file and prints some records."""
    print(f"\n--- Verifying final combined output file: {output_file_path} ---")
    try:
        if not os.path.exists(output_file_path):
            print(f"Output file not found at {output_file_path}.")
            return
        count = 0
        print(f"First {lines_to_show} records:\n")
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count < lines_to_show:
                    try:
                        if not line.strip(): continue
                        data = json.loads(line)
                        print(f"--- Record {count+1} ---")
                        print(f"  id          : {data.get('id')}")
                        print(f"  title       : {data.get('title', 'N/A')[:80]}...")
                        print(f"  categories  : {data.get('categories')}")
                        print(f"  abstract    : {data.get('abstract', '')[:100]}...")
                        print(f"  pdf_status  : {data.get('pdf_status')}")
                        full_text = data.get('full_text')
                        snippet = (full_text[:100].replace('\n', ' ') + "...") if isinstance(full_text, str) else "N/A"
                        print(f"  full_text   : {snippet}")
                        print("-" * 25)
                    except json.JSONDecodeError:
                        print(f"\n--- Record {count+1} (Corrupted JSON) ---")
                        print(line[:200] + "...")
                        print("-" * 25)
                count += 1
                if count >= lines_to_show: break
        print(f"\nDisplayed first {min(count, lines_to_show)} records found.")
    except Exception as e:
        print(f"Error reading verification file: {e}")
    print("----------------------------------------------------")


# --- Script Execution ---
if __name__ == "__main__":
    print("--- Local arXiv Data Processor (Parallel & Selective Output) ---")

    # --- Configuration Check ---
    print("\n--- Configuration ---")
    print(f"Input Metadata Path : {KAGGLE_METADATA_FILE}")
    print(f"Output Data Path    : {FINAL_OUTPUT_FILE}")
    print(f"Filter Categories   : ({len(FILTER_CATEGORIES)} specified)")
    target_str = str(MAX_PAPERS_TO_PROCESS) if MAX_PAPERS_TO_PROCESS is not None else "All Matching"
    print(f"Max NEW Papers Target: {target_str}")
    print(f"Number of Workers   : {NUM_WORKERS}")
    print(f"PDF Download Timeout: {REQUESTS_TIMEOUT}s")
    print(f"arXiv API Delay/Paper: {ARXIV_API_DELAY_PER_ID}s (IMPORTANT!)")
    runtime_str = f"{MAX_RUNTIME_SECONDS / 3600:.2f} hours" if MAX_RUNTIME_SECONDS is not None else "Unlimited"
    print(f"Max Runtime Limit   : {runtime_str}")
    print(f"Progress Log Interval: {LOG_INTERVAL_SECONDS} seconds")
    print(f"Log File Path       : {log_file_path}")
    print("----------------------")
    print("NOTE: Output records contain selected fields. Using parallel processing.")

    if not os.path.exists(KAGGLE_METADATA_FILE):
        print(f"\nERROR: Input metadata file '{KAGGLE_METADATA_FILE}' not found!", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Starting Parallel Data Collection/Merging Process ---")

    start_run_time_main = time.time()
    processed_ok_count = 0

    try:
        processed_ok_count = run_parallel_processing(
            kaggle_metadata_file=KAGGLE_METADATA_FILE,
            output_file=FINAL_OUTPUT_FILE,
            filter_categories=FILTER_CATEGORIES,
            max_papers_to_process_this_run=MAX_PAPERS_TO_PROCESS,
            fetch_timeout=REQUESTS_TIMEOUT,
            api_delay=ARXIV_API_DELAY_PER_ID,
            max_runtime_seconds=MAX_RUNTIME_SECONDS,
            log_interval_seconds=LOG_INTERVAL_SECONDS,
            num_workers=NUM_WORKERS
        )
    except Exception as e:
         logger.critical(f"Script execution failed critically outside processing loop: {e}", exc_info=True)
         print(f"A critical error occurred: {e}", file=sys.stderr)

    end_run_time_main = time.time()
    print(f"\n--- Main Process Complete ---")
    print(f"Total execution time: {end_run_time_main - start_run_time_main:.2f} seconds ({(end_run_time_main - start_run_time_main)/3600:.2f} hours).")
    print(f"Successfully processed and saved {processed_ok_count} new papers in this run (check logs for failures).")
    print(f"Combined data saved/appended to: {FINAL_OUTPUT_FILE}")
    print(f"Check the log file for details: {log_file_path}")

    verify_output_file(FINAL_OUTPUT_FILE)

    print("\nRun finished.")