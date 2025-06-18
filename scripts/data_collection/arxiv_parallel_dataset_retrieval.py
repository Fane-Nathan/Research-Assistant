# source_document_generator_from_search.py
# This script processes arXiv search queries to find relevant paper IDs,
# then fetches metadata and full text for those IDs.
# Its output (a JSONL file) serves as the source "feed" for subsequent
# fine-tuning data reward modeling pipelines.
# MODIFIED: Reads configuration from hybrid_search_rag.config.
# MODIFIED: Implements CLI arguments for overrides and operational modes.
# MODIFIED: Added "Dry Run" and "Force Re-process" modes.
# MODIFIED: Enhanced end-of-run summary.
# MODIFIED: Uses PROJECT_ROOT from the imported config module.
# MODIFIED: Addressed Pylance UndefinedVariable and AttributeAccessIssue errors.
# MODIFIED: Increased arxiv.Client page_size for discovery phase.
# MODIFIED (by user request): Added timeout for future.result() and more granular logging.

import os
import sys
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union 
import requests
import fitz
import json
import io
import re
import random
from tqdm import tqdm
import arxiv 
import tenacity
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import gc
import argparse

logger = logging.getLogger("ArxivDataCollector")

# --- Helper Functions ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]
def get_user_agent(): return random.choice(USER_AGENTS)

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

# --- Worker Function for Processing a Single Paper ID ---
def retry_if_network_error_or_server_issue(exception):
    if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exception, requests.exceptions.HTTPError):
        if exception.response.status_code in [500, 502, 503, 504]:
            return True
    return False

def process_single_paper_id(
    paper_id_str: str,
    api_delay: float,
    fetch_timeout: int,
    category_filters: List[str],
    retry_config: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:

    paper_id = paper_id_str.strip()
    original_input_id = paper_id_str

    if not paper_id:
        return (original_input_id, None, "Error: Empty ID string provided")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(retry_config['stop_after_attempt']),
        wait=tenacity.wait_exponential(
            multiplier=retry_config['wait_exponential_multiplier'],
            min=retry_config['wait_exponential_min_seconds'],
            max=retry_config['wait_exponential_max_seconds']
        ),
        retry=tenacity.retry_if_exception(retry_if_network_error_or_server_issue),
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO)
    )
    def download_pdf_with_retry(pdf_url: str, timeout: int, headers: Dict[str,str]) -> requests.Response:
        logger.debug(f"[{original_input_id}] Attempting to download PDF from: {pdf_url} (Timeout: {timeout}s)")
        pdf_response = requests.get(pdf_url, timeout=timeout, headers=headers, stream=True)
        pdf_response.raise_for_status()
        return pdf_response

    arxiv_client_worker = arxiv.Client(page_size=1, delay_seconds=0.1, num_retries=3)
    full_text: Optional[str] = None
    pdf_status: str = "Not Attempted"
    pdf_url_from_api: Optional[str] = None
    output_record: Optional[Dict[str, Any]] = None
    api_metadata_status: str = "API Not Called"

    try:
        actual_delay = api_delay + random.uniform(0, api_delay * 0.2)
        time.sleep(actual_delay)
        logger.debug(f"[{original_input_id}] Starting processing after {actual_delay:.2f}s delay")

        search_by_id = arxiv.Search(id_list=[paper_id], max_results=1)
        found_result: Optional[arxiv.Result] = None
        api_attempt_start = time.time()
        logger.debug(f"[{original_input_id}] Attempting arXiv API metadata lookup...")
        try:
            results_generator = arxiv_client_worker.results(search_by_id)
            found_result = next(results_generator, None)
            api_attempt_duration = time.time() - api_attempt_start
            logger.debug(f"[{original_input_id}] arXiv API metadata lookup complete in {api_attempt_duration:.2f}s.")
            if found_result:
                api_metadata_status = "Success"
                paper_id = found_result.get_short_id()
            else:
                api_metadata_status = "API Error: No results for ID"
                logger.warning(f"[{original_input_id}] API returned no results for ID.")
                return (original_input_id, None, api_metadata_status)
        except tenacity.RetryError as retry_err:
            api_metadata_status = "API Error: RetryError (arxiv lib)"
            logger.error(f"[{original_input_id}] arXiv API call failed after internal retries (arxiv lib): {retry_err}", exc_info=False)
            return (original_input_id, None, api_metadata_status)
        except Exception as api_err:
            api_metadata_status = f"API Error: {api_err.__class__.__name__}"
            logger.error(f"[{original_input_id}] arXiv API call failed: {api_err}", exc_info=False)
            return (original_input_id, None, api_metadata_status)

        if not found_result:
            logger.error(f"[{original_input_id}] Critical: found_result is None after API call block. Status: {api_metadata_status}")
            return (original_input_id, None, "Internal Error: found_result None post-API")

        paper_categories = found_result.categories if found_result.categories else []
        category_match = False
        if not category_filters:
            category_match = True
        elif paper_categories:
            if any(any(p_cat.startswith(f_cat) for f_cat in category_filters) for p_cat in paper_categories):
                category_match = True

        if not category_match:
            status_msg = f"Skipped: Category mismatch (ID: {paper_id}, Categories: {paper_categories})"
            logger.debug(f"[{paper_id}] {status_msg}")
            return (paper_id, None, status_msg)

        if found_result.pdf_url:
            pdf_url_from_api = found_result.pdf_url
            logger.debug(f"[{paper_id}] PDF URL from API: {pdf_url_from_api}. Attempting download...")
            headers = {'User-Agent': get_user_agent()}
            pdf_response_obj = None
            try:
                pdf_response_obj = download_pdf_with_retry(pdf_url_from_api, fetch_timeout, headers)
                logger.debug(f"[{paper_id}] PDF download HTTP request finished. Status: {pdf_response_obj.status_code}")
                content_type = pdf_response_obj.headers.get("Content-Type", "").lower()
                if 'application/pdf' in content_type:
                    try:
                        logger.debug(f"[{paper_id}] Reading PDF content bytes...")
                        pdf_bytes = pdf_response_obj.content
                        logger.debug(f"[{paper_id}] PDF content bytes read ({len(pdf_bytes)} bytes). Attempting parsing...")
                        if len(pdf_bytes) == 0:
                            logger.warning(f"[{paper_id}] Downloaded PDF is empty from {pdf_url_from_api}.")
                            pdf_status = "Download Error: Empty File"
                        else:
                            full_text = _parse_pdf_content(pdf_bytes, pdf_url_from_api)
                            pdf_status = "Success" if full_text else "Parsing Failed"
                        logger.debug(f"[{paper_id}] PDF parsing finished. Status: {pdf_status}")
                    except requests.exceptions.RequestException as read_err:
                         logger.error(f"[{paper_id}] Error reading downloaded PDF content from {pdf_url_from_api}: {read_err}. Using abstract.")
                         pdf_status = f"Download Error: Read Failure"
                else:
                    pdf_status = f"Wrong Content-Type: {content_type}"
                    logger.warning(f"[{paper_id}] PDF download from {pdf_url_from_api} yielded wrong content type: {content_type}. Using abstract.")
            except tenacity.RetryError as retry_err_pdf:
                 logger.error(f"[{paper_id}] PDF download failed after multiple retries for {pdf_url_from_api}: {retry_err_pdf}. Using abstract.")
                 pdf_status = f"Download Error: Retries Exhausted"
            except requests.exceptions.HTTPError as http_err:
                 logger.error(f"[{paper_id}] PDF download HTTP error for {pdf_url_from_api}: {http_err}. Using abstract.")
                 pdf_status = f"Download Error: HTTP {http_err.response.status_code}"
            except Exception as req_err:
                 logger.error(f"[{paper_id}] PDF download general error for {pdf_url_from_api}: {req_err}. Using abstract.")
                 pdf_status = f"Download Error: {req_err.__class__.__name__}"
            finally:
                if pdf_response_obj:
                    pdf_response_obj.close()
        else:
            pdf_status = "No PDF URL via API"
            logger.info(f"[{paper_id}] No PDF URL found via API. Using abstract if available.")

        if pdf_status != "Success":
            if found_result.summary:
                full_text = _clean_text(found_result.summary)
                if not full_text:
                    full_text = "Abstract available via API but empty after cleaning."
                    logger.warning(f"[{paper_id}] Abstract was empty after cleaning.")
                else:
                    logger.info(f"[{paper_id}] Used abstract because PDF status was: {pdf_status}.")
            else:
                full_text = "Text not available (PDF failed and no abstract from API)."
                logger.warning(f"[{paper_id}] No abstract available from API and PDF status was: {pdf_status}.")
                if pdf_status == "Not Attempted": pdf_status = "No Text Available"

        output_record = {
            'id': paper_id, 'title': found_result.title,
            'authors': [author.name for author in found_result.authors],
            'categories': found_result.categories, 'abstract': _clean_text(found_result.summary),
            'doi': found_result.doi, 'full_text': full_text,
            'pdf_fetched_url': pdf_url_from_api, 'pdf_status': pdf_status,
            'api_metadata_status': api_metadata_status
        }
        final_status_msg = f"API: {api_metadata_status}, PDF: {pdf_status}"
        return (paper_id, output_record, final_status_msg)

    except Exception as e:
        logger.error(f"[{original_input_id}] UNHANDLED error in worker: {e}", exc_info=True)
        return (original_input_id, None, f"Worker Error: {e.__class__.__name__}")
# --- End Worker Function ---

# --- Logger Setup Function ---
def setup_logger(log_level_str: str, project_root: str, log_dir_name: str, log_filename: str) -> str:
    global logger
    log_dir_full_path = os.path.join(project_root, log_dir_name)

    try:
        os.makedirs(log_dir_full_path, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create log directory '{log_dir_full_path}': {e}. Attempting to log in CWD/{log_dir_name}_fallback.", file=sys.stderr)
        log_dir_full_path = os.path.join(os.getcwd(), f"{log_dir_name}_fallback")
        try:
            os.makedirs(log_dir_full_path, exist_ok=True)
        except OSError as e_cwd:
            print(f"CRITICAL ERROR: Could not create log directory in CWD '{log_dir_full_path}': {e_cwd}. Logging to console only.", file=sys.stderr)
            logging.basicConfig(level=log_level_str.upper() if log_level_str else logging.INFO,
                                format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
            return "console_only"

    log_file_full_path = os.path.join(log_dir_full_path, log_filename)
    numeric_log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(numeric_log_level)
    log_formatter_obj = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s - %(funcName)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    try:
        file_handler_obj = logging.FileHandler(log_file_full_path, encoding='utf-8', mode='a')
        file_handler_obj.setFormatter(log_formatter_obj)
        logger.addHandler(file_handler_obj)
    except Exception as e_fh:
        print(f"Warning: Could not create file handler for '{log_file_full_path}': {e_fh}. Logging to console only.", file=sys.stderr)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            console_handler_obj = logging.StreamHandler(sys.stdout)
            console_handler_obj.setFormatter(log_formatter_obj)
            logger.addHandler(console_handler_obj)
        return "console_only_after_file_fail"

    console_handler_obj = logging.StreamHandler(sys.stdout)
    console_handler_obj.setFormatter(log_formatter_obj)
    logger.addHandler(console_handler_obj)
    logger.propagate = False

    return log_file_full_path

# --- Main Orchestration Function ---
def run_data_generation_pipeline(config: Dict[str, Any], dry_run: bool, force_reprocess: bool) -> Dict[str, Any]:
    logger.info(f"--- Starting Data Generation Pipeline ---")
    logger.info(f"Run Mode: {'Dry Run' if dry_run else 'Normal'}{' (Force Reprocess Enabled)' if force_reprocess and not dry_run else ''}")
    logger.info(f"Primary Target Categories: {config.get('primary_target_categories_for_discovery', 'N/A')}")
    logger.info(f"Max results per primary category: {config.get('max_results_per_primary_category_search', 'N/A')}")
    logger.info(f"Overall max papers to process: {config.get('max_papers_to_process', 'Unlimited')}")
    logger.info(f"Processing batch size: {config.get('processing_batch_size', 'N/A')}")
    logger.info(f"Using {config.get('num_workers', 'N/A')} worker threads. Outputting to: {config.get('output_source_documents_file', 'N/A')}")
    logger.info(f"Filter categories: {len(config.get('filter_categories', []))} specified")
    worker_timeout_seconds = config.get('worker_processing_timeout_seconds')
    logger.info(f"Individual paper processing timeout: {worker_timeout_seconds} seconds.")
    logger.info(f"---------------------------------------------------------------")

    stats = {
        'total_discovered_unique': 0,
        'already_processed_skipped': 0,
        'to_be_processed_this_run': 0,
        'processed_ok': 0, 'processed_fail': 0, 'category_mismatch_skipped': 0,
        'worker_timeouts': 0,
        'error_summary': {}
    }

    output_file = config['output_source_documents_file'] 
    primary_target_categories = config['primary_target_categories_for_discovery']
    max_results_per_primary_cat = config['max_results_per_primary_category_search']
    max_total_papers_to_process = config['max_papers_to_process'] 
    filter_categories_list = config['filter_categories']
    processing_batch_size = config['processing_batch_size']
    num_workers = config['num_workers']
    api_delay = config['arxiv_api_delay_per_id']
    fetch_timeout = config['requests_timeout']
    max_runtime_seconds = config['max_runtime_seconds']
    log_interval_seconds = config['progress_log_interval']

    retry_settings_for_worker = {
        'stop_after_attempt': config['retry_stop_after_attempt'],
        'wait_exponential_multiplier': config['retry_wait_exponential_multiplier'],
        'wait_exponential_min_seconds': config['retry_wait_exponential_min_seconds'],
        'wait_exponential_max_seconds': config['retry_wait_exponential_max_seconds']
    }

    if not output_file or not primary_target_categories: 
        logger.error("Essential configuration (output_file or primary_target_categories) is missing in effective_config. Cannot proceed.")
        return stats

    discovered_ids_to_consider = set()
    arxiv_search_client = arxiv.Client(
        page_size=config['arxiv_discovery_page_size'],
        delay_seconds=config['arxiv_discovery_delay_seconds'],
        num_retries=config['arxiv_discovery_num_retries']
    )

    logger.info("Phase 0: Discovering paper IDs from arXiv...")
    for cat_idx, target_category_code in enumerate(tqdm(primary_target_categories, desc="Searching Primary Categories")):
        search_query_for_category = f"cat:{target_category_code}"
        logger.info(f"Searching: '{search_query_for_category}' (Category {cat_idx+1}/{len(primary_target_categories)})")
        try:
            search = arxiv.Search(query=search_query_for_category, max_results=max_results_per_primary_cat,
                                  sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending)
            query_results_count = 0
            try:
                for result in arxiv_search_client.results(search):
                    discovered_ids_to_consider.add(result.get_short_id())
                    query_results_count += 1
                    if max_total_papers_to_process is not None and len(discovered_ids_to_consider) >= max_total_papers_to_process:
                        logger.info(f"Reached MAX_PAPERS_TO_PROCESS limit ({max_total_papers_to_process}) during discovery.")
                        break
            except arxiv.UnexpectedEmptyPageError as e:
                logger.warning(f"UnexpectedEmptyPageError for '{search_query_for_category}' after {query_results_count} results: {e}. Continuing discovery.")
            except tenacity.RetryError as e_retry:
                 logger.error(f"arXiv API search failed for '{search_query_for_category}' after internal retries (arxiv lib): {e_retry}. Continuing discovery.")
            logger.info(f"Found {query_results_count} results for '{search_query_for_category}'. Total unique IDs discovered so far: {len(discovered_ids_to_consider)}")
        except Exception as e:
            logger.error(f"Error during arXiv search for '{search_query_for_category}': {e}", exc_info=True)
        if max_total_papers_to_process is not None and len(discovered_ids_to_consider) >= max_total_papers_to_process:
            break

    unique_ids_to_process_list = list(discovered_ids_to_consider)
    random.shuffle(unique_ids_to_process_list)
    stats['total_discovered_unique'] = len(unique_ids_to_process_list)
    logger.info(f"Discovered {stats['total_discovered_unique']} unique paper IDs in total.")

    if max_total_papers_to_process is not None and len(unique_ids_to_process_list) > max_total_papers_to_process:
        unique_ids_to_process_list = unique_ids_to_process_list[:max_total_papers_to_process]
        logger.info(f"Limited to {len(unique_ids_to_process_list)} IDs due to MAX_PAPERS_TO_PROCESS config.")

    processed_ids_in_output = set()
    output_dir = os.path.dirname(output_file)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Ensured output directory exists: {output_dir}")
        except OSError as e:
            logger.critical(f"Could not create output directory '{output_dir}': {e}. Check permissions.", exc_info=True)
            return stats

    if not force_reprocess and os.path.exists(output_file):
        logger.info(f"Checking existing output file for resuming: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f_check:
                for line_num, line in enumerate(f_check):
                    try:
                        data = json.loads(line)
                        if 'id' in data: processed_ids_in_output.add(data['id'])
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping corrupted JSON line {line_num+1} in {output_file}")
        except Exception as e:
            logger.error(f"Error reading existing output file '{output_file}': {e}. Assuming no prior processed IDs.", exc_info=True)
            processed_ids_in_output = set()
    elif force_reprocess:
        logger.info(f"Force re-process ON. Contents of {output_file} will be overwritten (if processing starts).")

    final_ids_for_this_run = [pid for pid in unique_ids_to_process_list if pid not in processed_ids_in_output]
    stats['already_processed_skipped'] = len(unique_ids_to_process_list) - len(final_ids_for_this_run)
    stats['to_be_processed_this_run'] = len(final_ids_for_this_run)

    if dry_run:
        logger.info("--- Dry Run Summary ---")
        logger.info(f"Total unique paper IDs discovered: {stats['total_discovered_unique']}")
        if not force_reprocess: logger.info(f"Paper IDs in output (would be skipped): {stats['already_processed_skipped']}")
        logger.info(f"New paper IDs that WOULD BE processed: {stats['to_be_processed_this_run']}")
        return stats

    if not final_ids_for_this_run:
        logger.info("No new paper IDs to process. Exiting.")
        return stats

    start_run_time = time.time()
    last_log_time = start_run_time

    file_open_mode = 'w' if force_reprocess else 'a'
    if force_reprocess and os.path.exists(output_file) and final_ids_for_this_run:
        logger.warning(f"Output file '{output_file}' will be overwritten due to --force-reprocess.")
    elif force_reprocess and not os.path.exists(output_file) and final_ids_for_this_run:
        logger.info(f"Output file '{output_file}' will be created in write mode ('w') due to --force-reprocess.")

    try:
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='Worker') as executor, \
             open(output_file, file_open_mode, encoding='utf-8') as outfile_handle:

            total_batches = (len(final_ids_for_this_run) + processing_batch_size - 1) // processing_batch_size
            logger.info(f"Processing {len(final_ids_for_this_run)} IDs in {total_batches} batches of up to {processing_batch_size}.")

            for batch_num in range(total_batches):
                if max_runtime_seconds is not None and (time.time() - start_run_time) > max_runtime_seconds:
                    logger.warning("Reached MAX_RUNTIME_SECONDS limit before starting new batch.")
                    return stats

                batch_start_index = batch_num * processing_batch_size
                batch_end_index = batch_start_index + processing_batch_size
                current_batch_ids = final_ids_for_this_run[batch_start_index:batch_end_index]
                if not current_batch_ids: continue

                logger.info(f"Submitting Batch {batch_num + 1}/{total_batches} ({len(current_batch_ids)} IDs).")
                
                futures_to_ids = {}
                for pid_to_process in current_batch_ids:
                    future = executor.submit(process_single_paper_id, pid_to_process, api_delay, fetch_timeout, filter_categories_list, retry_settings_for_worker)
                    futures_to_ids[future] = pid_to_process

                if not futures_to_ids: continue

                for future in tqdm(as_completed(futures_to_ids), total=len(futures_to_ids), desc=f"Batch {batch_num+1}", unit="paper"):
                    original_paper_id_for_future = futures_to_ids[future]
                    try:
                        returned_paper_id, output_record, status_msg = future.result(timeout=worker_timeout_seconds)
                        
                        if output_record:
                            outfile_handle.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                            outfile_handle.flush()
                            stats['processed_ok'] += 1
                            del output_record
                        elif "Skipped: Category mismatch" in status_msg:
                            stats['category_mismatch_skipped'] += 1
                        else:
                            stats['processed_fail'] += 1
                            stats['error_summary'][status_msg] = stats['error_summary'].get(status_msg, 0) + 1
                        del status_msg
                    except FuturesTimeoutError:
                        logger.error(f"[{original_paper_id_for_future}] Worker timed out after {worker_timeout_seconds}s. Skipping this paper.")
                        stats['processed_fail'] += 1
                        stats['worker_timeouts'] += 1
                        error_key = f"WorkerTimeout ({worker_timeout_seconds}s)"
                        stats['error_summary'][error_key] = stats['error_summary'].get(error_key, 0) + 1
                    except Exception as e:
                        stats['processed_fail'] += 1
                        err_type = f"FutureResultError ({original_paper_id_for_future}): {e.__class__.__name__}"
                        logger.error(f"[{original_paper_id_for_future}] Error processing future: {e}", exc_info=False)
                        stats['error_summary'][err_type] = stats['error_summary'].get(err_type, 0) + 1

                    current_process_time = time.time()
                    if max_runtime_seconds is not None and (current_process_time - start_run_time) > max_runtime_seconds:
                        logger.warning("Reached MAX_RUNTIME_SECONDS limit during result processing. Shutting down.")
                        for f_cancel in futures_to_ids:
                            if not f_cancel.done(): f_cancel.cancel()
                        return stats

                    if current_process_time - last_log_time > log_interval_seconds:
                        elapsed_total_time = current_process_time - start_run_time
                        total_attempted_items_in_run = stats['processed_ok'] + stats['processed_fail'] + stats['category_mismatch_skipped']
                        
                        papers_per_sec = total_attempted_items_in_run / elapsed_total_time if elapsed_total_time > 0 else 0
                        remaining_to_process_count = len(final_ids_for_this_run) - total_attempted_items_in_run
                        eta_str = f"~{(remaining_to_process_count / papers_per_sec) / 3600:.1f}h" if papers_per_sec > 0 and remaining_to_process_count > 0 else "N/A"
                        
                        if max_runtime_seconds is not None:
                             remaining_runtime_sec = max(0, max_runtime_seconds - elapsed_total_time)
                             if papers_per_sec <= 0 or (remaining_to_process_count / papers_per_sec > remaining_runtime_sec):
                                  eta_str = f"~{remaining_runtime_sec / 3600:.1f}h (Runtime Limit)"
                        progress_runtime_str = f"{elapsed_total_time / 3600:.2f}/{max_runtime_seconds / 3600:.1f}h" if max_runtime_seconds is not None else f"{elapsed_total_time / 3600:.2f}h"
                        
                        if force_reprocess:
                            current_total_in_output_file = stats['processed_ok']
                        else:
                            current_total_in_output_file = len(processed_ids_in_output) + stats['processed_ok']
                        
                        logger.info(
                             f"Overall Progress: OK:{stats['processed_ok']} Fail:{stats['processed_fail']} (Timeouts:{stats['worker_timeouts']}) Cat.Skip:{stats['category_mismatch_skipped']} InitialOutputSkip:{stats['already_processed_skipped']} | "
                             f"Total in Output File (est.): {current_total_in_output_file} | "
                             f"Elapsed: {progress_runtime_str} | Rate: {papers_per_sec:.2f} papers/s | ETA: {eta_str}"
                        )
                        last_log_time = current_process_time
                
                if max_runtime_seconds is not None and (time.time() - start_run_time) > max_runtime_seconds:
                    logger.warning("Reached MAX_RUNTIME_SECONDS limit after batch completion.")
                    return stats

                logger.info(f"Batch {batch_num + 1} completed processing.")
                gc.collect()
    except IOError as io_err:
        logger.error(f"File I/O error with '{output_file}': {io_err}", exc_info=True)
        stats['error_summary']["IOError"] = stats['error_summary'].get("IOError", 0) + 1
    except KeyboardInterrupt:
         logger.warning("KeyboardInterrupt received. Shutting down gracefully...")
         stats['error_summary']["KeyboardInterrupt"] = stats['error_summary'].get("KeyboardInterrupt", 0) + 1
    except Exception as e:
         logger.error(f"Unexpected error in main pipeline execution: {e}", exc_info=True)
         stats['error_summary'][f"MainPipelineError: {e.__class__.__name__}"] = stats['error_summary'].get(f"MainPipelineError: {e.__class__.__name__}", 0) + 1
    return stats

# --- Verification Function (Unchanged) ---
def verify_output_file(output_file_path: str, lines_to_show: int = 5):
    print(f"\n--- Verifying source document output file: {output_file_path} ---")
    if not os.path.exists(output_file_path):
        print(f"Output file not found: {output_file_path}")
        return
    try:
        line_count = 0
        with open(output_file_path, 'r', encoding='utf-8') as f:
            print(f"First {lines_to_show} records (if available):\n")
            for i, line in enumerate(f):
                line_count +=1
                if i < lines_to_show:
                    print(line.strip())
        print(f"\nTotal records in file: {line_count}")
    except Exception as e:
        print(f"Error reading verification file: {e}")
    print("-" * 70)


# --- Main Execution Function ---
def main():
    parser = argparse.ArgumentParser(description="arXiv Source Document Generator with external config and CLI overrides.")
    # CLI arguments definitions remain the same
    parser.add_argument("--max-papers", type=int, help="Override ARXIV_PROC_MAX_PAPERS_TO_PROCESS.")
    parser.add_argument("--num-workers", type=int, help="Override ARXIV_PROC_NUM_WORKERS.")
    parser.add_argument("--batch-size", type=int, help="Override ARXIV_PROC_PROCESSING_BATCH_SIZE.")
    parser.add_argument("--output-filename", type=str, help="Override ARXIV_PROC_OUTPUT_FILENAME. Can be full path or relative to project_root/output_subdir.")
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Override ARXIV_PROC_LOG_LEVEL.")
    parser.add_argument("--dry-run", action="store_true", help="Discover IDs and show summary without processing.")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing all discovered/filtered papers, overwrites output file.")
    parser.add_argument("--worker-timeout", type=int, help="Override ARXIV_PROC_WORKER_PROCESSING_TIMEOUT_SECONDS (timeout for a single paper).")
    # Add CLI argument for path to config file directory
    parser.add_argument("--config-dir", type=str, help="Path to the directory containing the 'hybrid_search_rag' config package.")


    args = parser.parse_args()

    effective_config = {}
    project_config_module = None
    
    # Determine the directory for config import
    # Priority: CLI arg -> Environment Variable -> Script's guess
    config_package_parent_dir = args.config_dir
    
    if not config_package_parent_dir:
        config_package_parent_dir = os.getenv("HYBRID_RAG_CONFIG_DIR")
        if config_package_parent_dir:
            print(f"Using HYBRID_RAG_CONFIG_DIR environment variable for config path: {config_package_parent_dir}")

    if config_package_parent_dir:
        if not os.path.isdir(config_package_parent_dir):
            print(f"Error: Provided config directory '{config_package_parent_dir}' does not exist or is not a directory. Attempting script's default discovery...", file=sys.stderr)
            config_package_parent_dir = None
        else:
            print(f"Attempting to load config from specified directory: {config_package_parent_dir}")
            if config_package_parent_dir not in sys.path:
                sys.path.insert(0, config_package_parent_dir)
    
    # If config_package_parent_dir is still None, use script's original logic
    if not config_package_parent_dir:
        print("Config directory not specified via --config-dir or HYBRID_RAG_CONFIG_DIR. Using script's default discovery logic.")
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        config_package_parent_dir = os.path.dirname(os.path.dirname(current_script_dir))
        if config_package_parent_dir not in sys.path:
            sys.path.insert(0, config_package_parent_dir)
        print(f"Attempting to load config from script's determined parent directory: {config_package_parent_dir}")


    try:
        from hybrid_search_rag import config as imported_project_config
        project_config_module = imported_project_config
        # Use PROJECT_ROOT from the imported config module as the definitive one for path constructions
        # If not defined in config.py, use the directory that allowed the import, or a final fallback.
        effective_config['project_root'] = getattr(project_config_module, 'PROJECT_ROOT', config_package_parent_dir or os.getcwd())
        print(f"Successfully imported project configuration: hybrid_search_rag.config. PROJECT_ROOT set to: {effective_config['project_root']}")
    except ImportError as e:
        print(f"Warning: Could not import 'hybrid_search_rag.config': {e}. This may be due to incorrect path or missing __init__.py in 'hybrid_search_rag'.", file=sys.stderr)
        print(f"Checked sys.path includes: {config_package_parent_dir}", file=sys.stderr)
        effective_config['project_root'] = config_package_parent_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"Using fallback PROJECT_ROOT: {effective_config['project_root']}. Ensure this is correct for output/log paths.", file=sys.stderr)
    except Exception as e:
        print(f"CRITICAL ERROR during config import or PROJECT_ROOT setup: {e}", file=sys.stderr)
        effective_config['project_root'] = os.getcwd()
        print(f"Using CWD as PROJECT_ROOT due to error: {effective_config['project_root']}", file=sys.stderr)


    config_key_map = {
        "primary_target_categories_for_discovery": "ARXIV_PROC_PRIMARY_TARGET_CATEGORIES",
        "max_results_per_primary_category_search": "ARXIV_PROC_MAX_RESULTS_PER_PRIMARY_CATEGORY_SEARCH",
        "filter_categories": "ARXIV_PROC_FILTER_CATEGORIES",
        "max_papers_to_process": "ARXIV_PROC_MAX_PAPERS_TO_PROCESS",
        "num_workers": "ARXIV_PROC_NUM_WORKERS",
        "requests_timeout": "ARXIV_PROC_REQUESTS_TIMEOUT",
        "arxiv_api_delay_per_id": "ARXIV_PROC_API_DELAY_PER_ID",
        "processing_batch_size": "ARXIV_PROC_PROCESSING_BATCH_SIZE",
        "retry_stop_after_attempt": "ARXIV_PROC_RETRY_STOP_AFTER_ATTEMPT",
        "retry_wait_exponential_multiplier": "ARXIV_PROC_RETRY_WAIT_EXPONENTIAL_MULTIPLIER",
        "retry_wait_exponential_min_seconds": "ARXIV_PROC_RETRY_WAIT_EXPONENTIAL_MIN_SECONDS",
        "retry_wait_exponential_max_seconds": "ARXIV_PROC_RETRY_WAIT_EXPONENTIAL_MAX_SECONDS",
        "log_dir_name": "ARXIV_PROC_LOG_DIR_NAME",
        "log_filename": "ARXIV_PROC_LOG_FILENAME",
        "output_data_subdir": "ARXIV_PROC_OUTPUT_DATA_SUBDIR",
        "output_filename": "ARXIV_PROC_OUTPUT_FILENAME",
        "log_level": "ARXIV_PROC_LOG_LEVEL",
        "max_runtime_seconds": "ARXIV_PROC_MAX_RUNTIME_SECONDS",
        "progress_log_interval": "ARXIV_PROC_PROGRESS_LOG_INTERVAL",
        "worker_processing_timeout_seconds": "ARXIV_PROC_WORKER_PROCESSING_TIMEOUT_SECONDS",
        "arxiv_discovery_page_size": "ARXIV_DISCOVERY_PAGE_SIZE",
        "arxiv_discovery_delay_seconds": "ARXIV_DISCOVERY_DELAY_SECONDS",
        "arxiv_discovery_num_retries": "ARXIV_DISCOVERY_NUM_RETRIES",
    }
    
    # These defaults are ONLY used if project_config_module is None OR if a specific key is missing from it.
    script_defaults_for_fallback = {
        "primary_target_categories_for_discovery": [
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
            ],
        "max_results_per_primary_category_search": 5000,
        "filter_categories": [
            'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.CY',
            'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR',
            'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS',
            'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC',
            'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY',
            'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY',
            'stat.ML', 'stat.CO', 'stat.AP',
            'physics.app-ph', 'physics.ins-det', 'physics.optics', 'physics.comp-ph',
            'physics.data-an', 'physics.flu-dyn', 'physics.acc-ph', 'physics.plasm-ph', 'physics.space-ph',
            'physics.geo-ph', 'physics.med-ph',
            'math.IT', 'math.NA', 'math.OC', 'math.DS',
            'q-bio.CB', 'q-bio.GN', 'q-bio.QM', 'q-bio.BM', 'q-bio.SC', 'q-bio.TO',
            'q-fin.CP', 'q-fin.ST', 'q-fin.TR', 'q-fin.RM',
            'nlin.AO', 'nlin.PS',
            'quant-ph'
        ],
        "max_papers_to_process": 100000,
        "num_workers": 12,
        "requests_timeout": 300,
        "arxiv_api_delay_per_id": 0.25,
        "processing_batch_size": 10000,
        "retry_stop_after_attempt": 10,
        "retry_wait_exponential_multiplier": 1,
        "retry_wait_exponential_min_seconds": 5,
        "retry_wait_exponential_max_seconds": 30,
        "log_dir_name": "logs_arxiv_collection_fallback",
        "log_filename": "arxiv_collection_fallback.log",
        "output_data_subdir": "output_data_arxiv_fallback",
        "output_filename": "arxiv_default_output_fallback.jsonl",
        "log_level": "INFO",
        "max_runtime_seconds": None,
        "progress_log_interval": 300,
        "worker_processing_timeout_seconds": None,
        "arxiv_discovery_page_size": 5000,
        "arxiv_discovery_delay_seconds": 0.5,
        "arxiv_discovery_num_retries": 500,
    }

    for script_key, default_value in script_defaults_for_fallback.items():
        effective_config[script_key] = default_value 

    if project_config_module:
        print(f"Loading configurations from imported 'hybrid_search_rag.config' module.")
        for script_key, config_py_key in config_key_map.items():
            if hasattr(project_config_module, config_py_key):
                effective_config[script_key] = getattr(project_config_module, config_py_key)
            else:
                print(f"Info: Config key '{config_py_key}' for '{script_key}' not in config.py. Using script default: {effective_config[script_key]}")
    else:
        print("Warning: 'hybrid_search_rag.config' module not loaded. Using script defaults for all arXiv processing parameters.", file=sys.stderr)

    # Apply CLI overrides (highest precedence)
    if args.max_papers is not None: effective_config['max_papers_to_process'] = args.max_papers
    if args.num_workers is not None: effective_config['num_workers'] = args.num_workers
    if args.batch_size is not None: effective_config['processing_batch_size'] = args.batch_size
    if args.log_level: effective_config['log_level'] = args.log_level.upper()
    if args.worker_timeout is not None: effective_config['worker_processing_timeout_seconds'] = args.worker_timeout
    
    # Construct full output path using the now finalized 'project_root'
    # Note: args.output_filename can be an absolute path or a relative filename.
    # If it's relative, it's joined with project_root and the output_data_subdir from config.
    output_filename_cli = args.output_filename
    if output_filename_cli:
        if os.path.isabs(output_filename_cli):
            effective_config['output_source_documents_file'] = output_filename_cli
        else: # It's a relative filename, use it instead of the one from config
            effective_config['output_filename'] = output_filename_cli
            effective_config['output_source_documents_file'] = os.path.join(
                effective_config['project_root'],
                effective_config['output_data_subdir'], # subdir from config/defaults
                effective_config['output_filename']    # filename from CLI or config/defaults
            )
    else: # No CLI override for output_filename, use the one from config/defaults
        effective_config['output_source_documents_file'] = os.path.join(
            effective_config['project_root'],
            effective_config['output_data_subdir'],
            effective_config['output_filename']
        )
    
    log_file_full_path = setup_logger(
        effective_config['log_level'], 
        effective_config['project_root'], 
        effective_config['log_dir_name'], 
        effective_config['log_filename']
    )
    effective_config['log_file_path_full'] = log_file_full_path

    logger.info("--- arXiv Data Collector ---")
    logger.info(f"Using PROJECT_ROOT: {effective_config['project_root']}")
    logger.info(f"Effective Log Level: {effective_config['log_level']}")
    logger.info(f"Log File: {log_file_full_path}")
    logger.info(f"Output File: {effective_config['output_source_documents_file']}")
    
    print("\n--- Effective Configuration for this Run ---")
    for key in sorted(effective_config.keys()):
        value = effective_config[key]
        if key in ['filter_categories', 'primary_target_categories_for_discovery'] and isinstance(value, list):
            print(f"  {key:45s}: [{len(value)} categories specified]")
        else:
            print(f"  {key:45s}: {value}")
    print("--------------------------------------------")

    if args.force_reprocess:
        logger.warning("--force-reprocess is enabled. Output file will be overwritten if it exists and processing starts.")

    start_run_time_main = time.time()
    run_stats = {} 

    try:
        run_stats = run_data_generation_pipeline(
            config=effective_config,
            dry_run=args.dry_run,
            force_reprocess=args.force_reprocess
        )
    except Exception as e:
         logger.critical(f"Script execution failed critically in main: {e}", exc_info=True)
    finally:
        if not isinstance(run_stats, dict): 
            run_stats = {} 

        end_run_time_main = time.time()
        total_exec_time = end_run_time_main - start_run_time_main

        if args.dry_run: 
            logger.info("--- Dry Run Statistics (No actual processing occurred) ---")
            logger.info(f"Total execution time for dry run: {total_exec_time:.2f} seconds.")
            logger.info(f"Total unique paper IDs discovered: {run_stats.get('total_discovered_unique',0)}")
            if not args.force_reprocess:
                 logger.info(f"Paper IDs already in output (would be skipped): {run_stats.get('already_processed_skipped',0)}")
            logger.info(f"New paper IDs that WOULD BE processed: {run_stats.get('to_be_processed_this_run',0)}")
        else: 
            logger.info(f"\n--- Run Complete ---")
            logger.info(f"Total execution time: {total_exec_time:.2f} seconds ({total_exec_time/3600:.2f} hours).")
            logger.info(f"Successfully processed (OK): {run_stats.get('processed_ok', 0)}") 
            logger.info(f"Failed to process (Fail): {run_stats.get('processed_fail', 0)}")
            if run_stats.get('worker_timeouts', 0) > 0:
                logger.info(f"  -> Worker timeouts: {run_stats.get('worker_timeouts',0)}")
            logger.info(f"Category mismatches (Skipped): {run_stats.get('category_mismatch_skipped', 0)}")
            logger.info(f"Already processed in output (Skipped): {run_stats.get('already_processed_skipped',0)}")
            logger.info(f"Total considered for this run: {run_stats.get('to_be_processed_this_run',0)}")
            
            if run_stats.get('error_summary'):
                logger.info("Error Summary:")
                for err_msg, count in sorted(run_stats['error_summary'].items()):
                    logger.info(f"  - \"{err_msg}\": {count} times")

            logger.info(f"Source documents output to: {effective_config.get('output_source_documents_file', 'N/A')}")
            if os.path.exists(effective_config.get('output_source_documents_file','')): 
                 verify_output_file(effective_config['output_source_documents_file'])

        logger.info(f"Full log file available at: {effective_config.get('log_file_path_full', 'N/A')}")
        logger.info("Script finished.")

if __name__ == "__main__":
    main()