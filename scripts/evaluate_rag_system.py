"""
Command-Line Interface for Evaluating RAG Systems.

This script allows users to run the RAG evaluation pipeline from the command line.
It takes paths to an evaluation dataset, a Python module containing a RAG system
implementation, the RAG system class name, and an output path for the results.

Example Usage:
python scripts/evaluate_rag_system.py \
    --dataset_path /path/to/your/evaluation_dataset.json \
    --rag_module_path /path/to/your/rag_system_module.py \
    --rag_class_name YourRAGSystemClassName \
    --output_path /path/to/your/evaluation_results.json \
    --k_values 1,3,5,10 \
    --log_level INFO
"""

import argparse
import json
import logging
import os
import sys
import importlib.util
from datetime import datetime
from typing import List, Any, Type

# Adjust sys.path to include the project root for sibling module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hybrid_search_rag.evaluation.evaluation_dataset import EvaluationDataset
from hybrid_search_rag.evaluation.evaluation_engine import EvaluationRunner, RAGSystemInterface

logger = logging.getLogger(__name__)

def load_rag_system(module_path: str, class_name: str) -> RAGSystemInterface:
    """
    Dynamically loads a RAG system class from a given module path and instantiates it.

    Args:
        module_path: Absolute path to the Python file containing the RAG system class.
        class_name: The name of the RAG system class to load.

    Returns:
        An instance of the RAG system class.

    Raises:
        FileNotFoundError: If the module_path does not exist.
        AttributeError: If the class_name is not found in the module.
        TypeError: If the loaded class cannot be instantiated or does not conform
                   to the RAGSystemInterface (implicitly checked by EvaluationRunner).
    """
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"RAG system module not found at: {module_path}")

    module_name = os.path.splitext(os.path.basename(module_path))[0]
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from path: {module_path}")
    
    rag_module = importlib.util.module_from_spec(spec)
    # It's important to add the module to sys.modules BEFORE exec_module
    # if the module itself has internal relative imports.
    sys.modules[module_name] = rag_module 
    try:
        spec.loader.exec_module(rag_module)
    except Exception as e:
        logger.error(f"Error executing RAG module {module_path}: {e}", exc_info=True)
        raise

    if not hasattr(rag_module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module: {module_path}")
    
    RagSystemClass: Type[RAGSystemInterface] = getattr(rag_module, class_name)
    
    try:
        # Assuming the RAG system class can be instantiated without arguments.
        # If it needs arguments, this CLI would need to be extended.
        rag_system_instance = RagSystemClass()
        logger.info(f"Successfully loaded and instantiated RAG system '{class_name}' from '{module_path}'.")
        return rag_system_instance
    except Exception as e:
        logger.error(f"Error instantiating RAG system class '{class_name}': {e}", exc_info=True)
        raise TypeError(f"Could not instantiate RAG system class '{class_name}': {e}")

def main():
    """Main function to parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description="Run RAG System Evaluation")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing the evaluation dataset."
    )
    parser.add_argument(
        "--rag_module_path", 
        type=str, 
        required=True, 
        help="Path to the Python module file implementing the RAG system (RAGSystemInterface)."
    )
    parser.add_argument(
        "--rag_class_name", 
        type=str, 
        required=True, 
        help="Name of the RAG system class within the specified module."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save the evaluation results JSON file. Defaults to 'evaluation_results_<timestamp>.json'."
    )
    parser.add_argument(
        "--k_values", 
        type=str, 
        default="1,3,5,10", 
        help="Comma-separated list of integers for K in metrics (e.g., '1,5,10')."
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level."
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
    )

    logger.info("Starting RAG system evaluation script...")
    logger.info(f"Arguments: {args}")

    # Parse k_values
    try:
        k_values = [int(k.strip()) for k in args.k_values.split(',') if k.strip()]
        if not k_values or any(k <= 0 for k in k_values):
            raise ValueError("K-values must be positive integers.")
    except ValueError as e:
        logger.error(f"Invalid k_values format: '{args.k_values}'. {e}")
        sys.exit(1)

    # Determine output path
    output_path = args.output_path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_results_{timestamp}.json"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    try:
        # 1. Load Evaluation Dataset
        logger.info(f"Loading evaluation dataset from: {args.dataset_path}")
        if not os.path.exists(args.dataset_path):
            logger.error(f"Dataset file not found: {args.dataset_path}")
            sys.exit(1)
        eval_dataset = EvaluationDataset.load_from_json(args.dataset_path)
        logger.info(f"Loaded dataset '{eval_dataset.dataset_name}' with {len(eval_dataset.queries)} queries.")

        # 2. Load RAG System
        logger.info(f"Loading RAG system '{args.rag_class_name}' from module '{args.rag_module_path}'.")
        rag_system = load_rag_system(os.path.abspath(args.rag_module_path), args.rag_class_name)
        
        # 3. Initialize EvaluationRunner
        logger.info(f"Initializing EvaluationRunner with k_values: {k_values}")
        runner = EvaluationRunner(rag_system=rag_system, k_values=k_values)

        # 4. Run Evaluation
        logger.info("Starting evaluation run...")
        results = runner.run_evaluation(eval_dataset)
        logger.info("Evaluation run completed.")

        # 5. Save Results
        logger.info(f"Saving evaluation results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results successfully saved to {output_path}")
        
        # Print summary
        print("\n--- Evaluation Summary ---")
        print(f"Dataset: {results.get('dataset_name')}")
        print(f"Queries Processed: {results.get('num_queries_processed')}")
        if 'retrieval' in results.get('summary_metrics', {}):
            print("Retrieval Metrics:")
            for metric, value in results['summary_metrics']['retrieval'].items():
                if isinstance(value, dict):
                    print(f"  {metric}:")
                    for sub_metric, sub_value in value.items():
                        print(f"    {sub_metric}: {sub_value:.4f}")
                else:
                    print(f"  {metric}: {value:.4f}")
        print(f"Detailed results saved to: {os.path.abspath(output_path)}")
        print("--- End of Summary ---")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except AttributeError as e:
        logger.error(f"Attribute error (e.g., class not found in module): {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Import error (e.g., could not load RAG module): {e}")
        sys.exit(1)
    except TypeError as e:
        logger.error(f"Type error (e.g., RAG class instantiation issue or interface mismatch): {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
