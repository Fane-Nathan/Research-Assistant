# scripts/interactive_evaluation_labeler.py
"""
This script provides an interactive way to create the 'evaluation_dataset.json'
by guiding the user through labeling candidates from a pre-generated review file.
It can optionally use an LLM to suggest relevance scores.
"""
import sys
import os
import json
import logging
import re
from typing import List, Dict, Any, Set, Union, Optional

# --- Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

from hybrid_search_rag import config
# Import the LLM interface
from hybrid_search_rag.llm_services.llm_interface import get_llm_response


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_llm_suggested_relevance(query_text: str, doc_title: str, doc_snippet: str) -> Optional[int]:
    """
    Uses an LLM to suggest a relevance score for a document snippet given a query.
    """
    # Simplified, more open-ended prompt
    prompt = f"""You are an expert relevance assessor.
Query: "{query_text}"
Document Title: "{doc_title}"
Document Snippet: "{doc_snippet}"

Briefly assess the relevance of this document snippet to the query.
Then, on a new line, state a numerical relevance score from 0 to 3.
0: Not relevant.
1: Marginally relevant.
2: Relevant.
3: Highly relevant.

Example response format:
The snippet seems quite relevant as it discusses X.
Score: 2

Your assessment:"""
    try:
        logger.debug(f"Sending prompt to LLM for relevance suggestion: Query='{query_text[:50]}...', Title='{doc_title[:50]}...'")
        # Keep max_output_tokens generous for this more open prompt
        llm_suggestion_str = get_llm_response(prompt, generation_args={"max_output_tokens": 75})

        logger.info(f"Raw LLM suggestion string: '{llm_suggestion_str}'") # Crucial log

        if llm_suggestion_str and llm_suggestion_str.strip():
            # Try to find a line that starts with "Score: " followed by a digit
            match = re.search(r"Score:\s*([0-3])", llm_suggestion_str, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                logger.info(f"LLM suggested relevance (parsed from 'Score: X'): {score}")
                return score
            else:
                # Fallback: try to find any standalone digit 0-3 if "Score: X" isn't found
                match_digit_only = re.search(r'\b([0-3])\b', llm_suggestion_str)
                if match_digit_only:
                    score = int(match_digit_only.group(1))
                    logger.info(f"LLM suggested relevance (parsed as standalone digit): {score}")
                    return score
                else:
                    logger.warning(f"Could not parse 'Score: X' or a standalone digit (0-3) from LLM suggestion: '{llm_suggestion_str}'")
        else:
            logger.warning(f"LLM returned an empty or whitespace-only suggestion. Raw: '{llm_suggestion_str}'")
        return None
    except Exception as e:
        logger.error(f"Error getting LLM suggestion: {e}", exc_info=True)
        return None


def get_user_relevance_score(
    query_text: str,
    candidate: Dict[str, Any],
    candidate_idx: int,
    total_candidates: int,
    enable_llm_suggestions: bool = True
) -> Union[int, str]:
    """
    Prompts the user to enter a relevance score for a given candidate.
    Optionally shows an LLM suggestion. Allows skipping or exiting.
    """
    print("-" * 80)
    print(f"QUERY: {query_text}")
    print("-" * 80)
    print(f"CANDIDATE {candidate_idx + 1}/{total_candidates}:")
    print(f"  ID:       {candidate.get('doc_id')}")
    print(f"  Title:    {candidate.get('doc_title')}")
    print(f"  Snippet:  {candidate.get('doc_snippet')}")
    print(f"  Heuristic Score: {candidate.get('auto_heuristic_score', 'N/A'):.4f}" if isinstance(candidate.get('auto_heuristic_score'), float) else "")

    llm_suggested_score: Optional[int] = None
    if enable_llm_suggestions:
        print("  Getting LLM suggestion...")
        llm_suggested_score = get_llm_suggested_relevance(
            query_text,
            candidate.get('doc_title', ''),
            candidate.get('doc_snippet', '')
        )
        if llm_suggested_score is not None:
            print(f"  LLM Suggests Relevance: {llm_suggested_score}")
        else:
            print("  LLM suggestion not available or failed.")
    print("-" * 80)

    prompt_message = (
        "Enter relevance (0=Not, 1=Marginal, 2=Relevant, 3=Highly Relevant)"
    )
    if llm_suggested_score is not None:
        prompt_message += f" | (k)eep LLM suggestion ({llm_suggested_score})"

    prompt_message += " | (s)kip doc | (sq)skip query | (f)inish query | (x)exit & save: "


    while True:
        try:
            user_input = input(prompt_message).strip().lower()

            if user_input in ['0', '1', '2', '3']:
                return int(user_input)
            elif user_input in ['k', 'keep'] and llm_suggested_score is not None:
                logger.info(f"User kept LLM suggestion: {llm_suggested_score}")
                return llm_suggested_score
            elif user_input in ['s', 'skip']:
                return 'skip_doc'
            elif user_input in ['sq', 'skip_query']:
                return 'skip_query'
            elif user_input in ['f', 'finish', 'finish_query']:
                return 'finish_query'
            elif user_input in ['x', 'exit']:
                return 'exit_labelling'
            else:
                print("Invalid input. Please enter a number 0-3, or k, s, sq, f, x.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            logger.warning("\nLabelling interrupted by user. Exiting without saving current query progress.")
            return 'exit_labelling'

def interactive_labeling(
    input_candidates_file_path: str,
    output_evaluation_dataset_path: str,
    resume_from_query_id: Optional[str] = None,
    use_llm_suggestions: bool = True # Control LLM suggestions
):
    """
    Interactively labels candidates to create the evaluation_dataset.json.
    """
    logger.info(f"Loading candidates for labeling from: {input_candidates_file_path}")
    try:
        with open(input_candidates_file_path, 'r', encoding='utf-8') as f:
            all_queries_with_candidates = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_candidates_file_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_candidates_file_path}: {e}")
        return

    final_evaluation_data = []
    if os.path.exists(output_evaluation_dataset_path):
        logger.info(f"Found existing evaluation dataset at: {output_evaluation_dataset_path}. Will try to resume/append.")
        try:
            with open(output_evaluation_dataset_path, 'r', encoding='utf-8') as f_existing:
                final_evaluation_data = json.load(f_existing)
                logger.info(f"Loaded {len(final_evaluation_data)} already labeled queries.")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not load or parse existing output file, starting fresh: {e}")
            final_evaluation_data = []

    labeled_query_ids = {item['query_id'] for item in final_evaluation_data}
    
    queries_to_process_initially = [q for q in all_queries_with_candidates if q['query_id'] not in labeled_query_ids]

    if resume_from_query_id:
        try:
            start_index = next(i for i, q in enumerate(queries_to_process_initially) if q['query_id'] == resume_from_query_id)
            queries_to_process = queries_to_process_initially[start_index:]
            logger.info(f"Resuming labeling from query_id: {resume_from_query_id} (index {start_index} in remaining queries).")
        except StopIteration:
            logger.warning(f"Resume query_id '{resume_from_query_id}' not found in remaining queries. Starting from the beginning of remaining unlabeled queries.")
            queries_to_process = queries_to_process_initially
    else:
        queries_to_process = queries_to_process_initially


    if not queries_to_process:
        logger.info("All queries from the input file seem to have been labeled already or no new queries to process.")

    total_queries_to_label = len(queries_to_process)
    logger.info(f"Starting interactive labeling for {total_queries_to_label} queries. LLM suggestions: {'Enabled' if use_llm_suggestions else 'Disabled'}")

    for query_idx, query_item in enumerate(queries_to_process):
        clear_screen()
        query_id = query_item.get("query_id")
        query_text = query_item.get("query_text", "")
        candidates = query_item.get("suggested_candidates_for_review", [])

        if not candidates:
            logger.info(f"Query '{query_id}' has no candidates to label. Skipping.")
            continue

        print(f"\n--- Query {query_idx + 1}/{total_queries_to_label} ({query_id}) ---")
        current_query_relevance_scores: Dict[str, float] = {}
        current_query_relevant_doc_ids: List[str] = []

        for cand_idx, candidate in enumerate(candidates):
            doc_id = candidate.get("doc_id")
            if not doc_id:
                logger.warning("Candidate missing 'doc_id'. Skipping this candidate.")
                continue

            clear_screen()
            action = get_user_relevance_score(
                query_text,
                candidate,
                cand_idx,
                len(candidates),
                enable_llm_suggestions=use_llm_suggestions
            )

            if isinstance(action, int):
                score = float(action)
                current_query_relevance_scores[doc_id] = score
                if score > 0:
                    current_query_relevant_doc_ids.append(doc_id)
            elif action == 'skip_doc':
                logger.info(f"Skipping document {doc_id} for query {query_id}.")
                continue
            elif action == 'skip_query':
                logger.info(f"Skipping query {query_id}.")
                current_query_relevance_scores.clear()
                current_query_relevant_doc_ids.clear()
                break
            elif action == 'finish_query':
                logger.info(f"Finished labeling for query {query_id}.")
                break
            elif action == 'exit_labelling':
                logger.info("Exiting labeling process. Saving current progress...")
                if current_query_relevance_scores:
                    existing_entry_index = next((i for i, item in enumerate(final_evaluation_data) if item['query_id'] == query_id), -1)
                    new_entry = {
                        "query_id": query_id,
                        "query_text": query_text,
                        "relevant_doc_ids": list(set(current_query_relevant_doc_ids)),
                        "relevance_scores": current_query_relevance_scores
                    }
                    if existing_entry_index != -1:
                        final_evaluation_data[existing_entry_index] = new_entry
                    else:
                        final_evaluation_data.append(new_entry)
                _save_evaluation_data(output_evaluation_dataset_path, final_evaluation_data)
                return

        if current_query_relevance_scores: # Save if any labels were given for this query
            existing_entry_index = next((i for i, item in enumerate(final_evaluation_data) if item['query_id'] == query_id), -1)
            new_entry = {
                "query_id": query_id,
                "query_text": query_text,
                "relevant_doc_ids": list(set(current_query_relevant_doc_ids)), # Ensure uniqueness
                "relevance_scores": current_query_relevance_scores
            }
            if existing_entry_index != -1:
                logger.info(f"Updating existing entry for query_id: {query_id}")
                final_evaluation_data[existing_entry_index] = new_entry
            else:
                final_evaluation_data.append(new_entry)
            _save_evaluation_data(output_evaluation_dataset_path, final_evaluation_data)

    logger.info("Interactive labeling complete.")
    _save_evaluation_data(output_evaluation_dataset_path, final_evaluation_data) # Final save

def _save_evaluation_data(file_path: str, data: List[Dict[str, Any]]):
    """Helper function to save the evaluation data to JSON."""
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation data saved to: {file_path} ({len(data)} queries)")
    except Exception as e:
        logger.error(f"Error saving evaluation data to {file_path}: {e}", exc_info=True)

if __name__ == '__main__':
    candidates_file = os.path.join(project_root, "data_store", "evaluation_sets", "scored_candidates_for_review.json")
    if not os.path.exists(candidates_file):
        logger.info(f"'{candidates_file}' not found. Trying 'candidates_for_review.json' instead.")
        candidates_file = os.path.join(project_root, "data_store", "evaluation_sets", "candidates_for_review.json")

    evaluation_output_file = os.path.join(project_root, "data_store", "evaluation_sets", "evaluation_dataset.json")

    ENABLE_LLM_ASSISTANCE = True
    resume_query = None

    if not os.path.exists(candidates_file):
        logger.error(f"Input candidates file not found at either expected location: {candidates_file}")
        logger.error("Please run 'bootstrap_evaluation_set.py' and optionally 'score_review_candidates.py' first.")
    else:
        interactive_labeling(
            candidates_file,
            evaluation_output_file,
            resume_from_query_id=resume_query,
            use_llm_suggestions=ENABLE_LLM_ASSISTANCE
        )
