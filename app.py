# -*- coding: utf-8 -*-
"""
Streamlit web application for the Study Assistant (User Provided Version).

Provides an enhanced user interface for:
1. Fetching and processing resources (arXiv, web, potentially others via cli functions).
2. Performing hybrid search and RAG to answer questions based on local data.
3. Directly searching arXiv.
4. Displaying information about the project and how it works.
5. Collecting user feedback via an embedded form.
6. Adding content by crawling a single user-provided URL.
"""

import json
import streamlit as st
# --- Page Configuration MUST be the first Streamlit command ---
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/Fane-Nathan/Research-Assistant/tree/alpha-release',
        'Report a bug': "https://tally.so/r/n0kkp6",
        'About': "# About This Project\nThis app helps explore academic research using RAG."
    }
)

# --- Other Imports ---
import logging
import os
import sys
import asyncio
import types # Add this import
import argparse # Add this import
import streamlit.components.v1 as components # Add this import

from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional # Ensure Any is imported
import numpy as np # Keep if used by components

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [App-Console] %(message)s',
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# --- Path Setup ---
logger.info("Setting up sys.path...")
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project Root added to sys.path: {project_root}")

# --- Project Module Imports ---
logger.info("Attempting project module imports...")
project_modules_loaded = False
config = None
check_nltk_data_cli = None # Renamed to avoid conflict if any local check_nltk_data
load_components_cli = None # Renamed
setup_data_and_fetch_cli = None # Renamed
run_arxiv_search_cli = None # Renamed
cli_run_recommendation_async = None # Renamed

try:
    from hybrid_search_rag import config
    from scripts.cli import (
        load_components as load_components_cli, # Import the modified function
        setup_data_and_fetch as setup_data_and_fetch_cli,
        run_arxiv_search as run_arxiv_search_cli,
        run_recommendation as cli_run_recommendation_async, # Corrected import: use run_recommendation
        check_nltk_data as check_nltk_data_cli,
        chunk_text_by_sentences # Keep if used directly in app.py, otherwise can be removed
    )
    # DataManager, ResourceFetcher, EmbeddingModel, NltkManager, HybridRecommender
    # are now primarily managed within scripts.cli.load_components
    project_modules_loaded = True
    logger.info("Project module imports successful.")
except ImportError as e:
    st.error("🔧 **System setup incomplete.** Please ensure all required components are properly installed and refresh the page. If the issue persists, check the installation guide or contact support.", icon="📋")
    st.code(f"Current sys.path: {sys.path}")
    logger.error(f"Project import failed (ImportError): {e}", exc_info=True)
except Exception as e:
    st.error("🔧 **Startup issue detected.** Some features may not work correctly. Please refresh the page or contact support if the problem persists.", icon="⚠️")
    logger.error(f"Project import failed (Exception): {e}", exc_info=True)

# --- Windows asyncio policy fix ---
if sys.platform == "win32" and "asyncio" in sys.modules: # Check if asyncio is imported
    logger.info("Applying WindowsProactorEventLoopPolicy for asyncio.")
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# --- Centralized Component Initialization and State Management ---
def initialize_and_load_components():
    """
    Initializes and loads RAG components, updating session state.
    This function is designed to be called on app start and after data updates.
    """
    logger.info("Attempting to initialize and load RAG components...")
    if not project_modules_loaded or not load_components_cli:
        logger.error("Cannot initialize components: project modules or load_components_cli not available.")
        st.session_state.components_loaded_successfully = False
        st.session_state.rag_components = None
        st.session_state.knowledge_base_is_empty = True # Default if critical modules missing
        st.session_state.component_status_message = "Error: Core application modules not loaded."
        return

    # Ensure NLTK data is checked before loading components
    if check_nltk_data_cli:
        if not check_nltk_data_cli(): # check_nltk_data_cli from scripts.cli
            logger.error("NLTK data check failed. Components might not load correctly.")
            # Optionally, set a more specific error state or message
            # For now, load_components_cli will also log issues.
    else:
        logger.warning("check_nltk_data_cli function not available.")

    force_reload = st.session_state.get("force_component_reload", False)
    logger.info(f"Calling load_components_cli with force_reload={force_reload}")
    
    loaded_data_dict = load_components_cli(force_reload=force_reload)
    st.session_state.force_component_reload = False # Reset flag

    if loaded_data_dict and loaded_data_dict.get("recommender"):
        st.session_state.rag_components = loaded_data_dict
        st.session_state.components_loaded_successfully = True
        data_manager = loaded_data_dict.get("data_manager")
        if data_manager and hasattr(data_manager, 'is_data_empty') and callable(data_manager.is_data_empty):
            st.session_state.knowledge_base_is_empty = data_manager.is_data_empty()
        else: # Fallback if DataManager or is_data_empty is missing
            logger.warning("DataManager or is_data_empty method not available from load_components_cli. Performing direct file check for KB status.")
            # Direct file check as a robust fallback
            kb_files_found = False
            if config and config.DATA_DIR and config.METADATA_FILE:
                metadata_file_path = os.path.join(config.DATA_DIR, config.METADATA_FILE)
                if os.path.exists(metadata_file_path) and os.path.getsize(metadata_file_path) > 10: # Check if file exists and is not empty
                    try:
                        with open(metadata_file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            if isinstance(content, list) and len(content) > 0:
                                kb_files_found = True
                    except json.JSONDecodeError:
                        logger.error(f"Could not parse metadata file at {metadata_file_path} during fallback check.")
                    except Exception as e_fallback:
                        logger.error(f"Error during fallback KB check: {e_fallback}")

            st.session_state.knowledge_base_is_empty = not kb_files_found
        st.session_state.component_status_message = loaded_data_dict.get("status_message", "Components loaded.")
        logger.info(f"Components initialized successfully. KB empty: {st.session_state.knowledge_base_is_empty}. Status: {st.session_state.component_status_message}")
    else:
        logger.warning(f"Component initialization failed or recommender not loaded. Status: {loaded_data_dict.get('status_message', 'Unknown error')}")
        st.session_state.rag_components = loaded_data_dict # Store whatever was returned for debugging
        st.session_state.components_loaded_successfully = False
        # Fallback KB check if components failed to load properly
        kb_files_found = False
        if config and config.DATA_DIR and config.METADATA_FILE:
            metadata_file_path = os.path.join(config.DATA_DIR, config.METADATA_FILE)
            if os.path.exists(metadata_file_path) and os.path.getsize(metadata_file_path) > 10:
                 try:
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        if isinstance(content, list) and len(content) > 0:
                            kb_files_found = True
                 except Exception: pass # Ignore errors here, just trying to check
        st.session_state.knowledge_base_is_empty = not kb_files_found
        st.session_state.component_status_message = loaded_data_dict.get("status_message", "Components failed to load.")
        logger.info(f"Component initialization failed. Fallback KB check - KB empty: {st.session_state.knowledge_base_is_empty}. Status: {st.session_state.component_status_message}")


async def handle_recommendation_submission_async(
    query_text: str,
    top_n_results: int,
    effective_general_mode: bool,
    use_concise_mode: bool,
    answer_placeholder: Any, # Use typing.Any for broader compatibility
    # project_modules_loaded is globally available, no need to pass if check is outside
):
    """
    Handles the asynchronous submission of a recommendation request.
    Calls the CLI's recommendation function, streams the answer to the UI,
    updates session state, and handles errors.
    """
    st.session_state.recommendation_processing = True
    st.session_state.llm_answer = ""  # Initialize for accumulation/display
    st.session_state.context_sources = []
    st.session_state.raw_context_chunks = [] # Ensure this is initialized
    current_answer_display = ""

    try:
        logger.info(f"handle_recommendation_submission_async: Calling cli_run_recommendation_async for query '{query_text[:50]}...'")
        
        # Ensure cli_run_recommendation_async is available (it's imported globally)
        if not cli_run_recommendation_async:
            logger.error("cli_run_recommendation_async is not available!")
            st.session_state.llm_answer = "Error: Recommendation function not loaded."
            answer_placeholder.error(st.session_state.llm_answer)
            return # Exit early

        result = await cli_run_recommendation_async(
            query=query_text,
            num_final_results=top_n_results, # Changed top_n to num_final_results
            general_mode=effective_general_mode,
            concise_mode=use_concise_mode,
            # rag_components are accessed by cli_run_recommendation_async internally from st.session_state
        )

        if result:
            answer_data, sources_data, raw_chunks_data = result

            st.session_state.context_sources = sources_data if sources_data else []
            st.session_state.raw_context_chunks = raw_chunks_data if raw_chunks_data else []

            if isinstance(answer_data, types.GeneratorType): # Changed from AsyncGenerator
                for token in answer_data: # Changed from 'async for'
                    current_answer_display += token
                    answer_placeholder.markdown(current_answer_display + "▌")  # Display with a cursor
                st.session_state.llm_answer = current_answer_display
                if current_answer_display:
                    answer_placeholder.markdown(current_answer_display) # Final display
                else:
                    st.session_state.llm_answer = "No specific answer generated from the provided context or query."
                    answer_placeholder.info(st.session_state.llm_answer)
            elif isinstance(answer_data, str):
                st.session_state.llm_answer = answer_data
                if answer_data:
                    answer_placeholder.markdown(answer_data)
                else:
                    st.session_state.llm_answer = "No specific answer generated."
                    answer_placeholder.info(st.session_state.llm_answer)
            else:
                error_msg = f"Unexpected answer data type from RAG: {type(answer_data)}"
                logger.error(error_msg)
                st.session_state.llm_answer = "Error: Could not process the AI's response format."
                answer_placeholder.error(st.session_state.llm_answer)
        else:
            logger.warning("cli_run_recommendation_async returned no result.")
            st.session_state.llm_answer = "No recommendation could be generated for this query."
            answer_placeholder.info(st.session_state.llm_answer)

    except Exception as e:
        detailed_error_msg = f"Error during recommendation processing: {e}"
        logger.error(detailed_error_msg, exc_info=True)
        st.session_state.llm_answer = f"🤖 **An error occurred while generating the recommendation:** {str(e)[:200]}..."
        if answer_placeholder:
            answer_placeholder.error(st.session_state.llm_answer)
    finally:
        st.session_state.recommendation_processing = False
        st.rerun()


# --- Initial NLTK Data Check (once per session) ---
if 'nltk_data_checked_app' not in st.session_state:
    if project_modules_loaded and check_nltk_data_cli:
        with st.spinner("Verifying language processing components..."):
            logger.info("APP.PY: Running initial NLTK check via imported cli.check_nltk_data_cli...")
            check_nltk_data_cli()  # From scripts.cli
            st.session_state.nltk_data_checked_app = True
            logger.info("APP.PY: Initial NLTK check complete.")
    elif not project_modules_loaded:
        logger.warning("APP.PY: NLTK check skipped: project_modules_loaded is False.")
    else: # project_modules_loaded is True, but check_nltk_data_cli is not
        logger.warning("APP.PY: NLTK check skipped: check_nltk_data_cli function not available.")
        st.warning("🔧 **Setup Incomplete:** Core language components (NLTK check function) are not available. Please wait or refresh.", icon="⏳")


# --- Initial Component Load (once per session or after forced reload) ---
if 'components_loaded_successfully' not in st.session_state:
    st.session_state.force_component_reload = True # Ensure first load is fresh
    initialize_and_load_components()
    st.session_state.force_component_reload = False # Reset after initial load

# Log current status after potential initialization
logger.info(f"App Rerun. Components loaded: {st.session_state.get('components_loaded_successfully', False)}, KB empty: {st.session_state.get('knowledge_base_is_empty', True)}")
logger.info(f"Component status message: {st.session_state.get('component_status_message', 'Not set')}")


# --- Title, Introduction, and Disclaimer ---
st.title("📚 Research Paper Assistant")
st.subheader("Unlock insights from academic literature with AI")
st.markdown("""
Explore academic research effortlessly. This tool leverages Retrieval-Augmented Generation (RAG)
and Hybrid Search to find relevant papers and answer your questions using document context.
""")
with st.expander("ℹ️ Important Disclaimer", expanded=False):
    st.warning("""
    **Please Note:** This tool is specialized for finding and analyzing information within the indexed research papers.
    It excels at providing context-aware answers based on its knowledge base.

    It is **not** a general-purpose chatbot (like ChatGPT, Gemini, etc.) and may not perform well on:
    * General knowledge questions outside the scope of the indexed documents.
    * Creative writing or conversational tasks.
    * Real-time information (e.g., news, weather).
    """, icon="⚠️")
st.divider()

# --- Welcome Message / Status ---
if not st.session_state.get('components_loaded_successfully', False) and project_modules_loaded:
    if st.session_state.get('knowledge_base_is_empty', True):
        st.info("🎯 **Welcome to Research Assistant!** To get started, please add some research papers to the knowledge base using the **'Update Knowledge Base'** tab below. This will enable the AI recommendation features.")
        st.markdown("💡 **Tip:** Try searching for topics like 'machine learning', 'quantum computing', or 'artificial intelligence' to build your initial knowledge base.")
    else: # Files might exist, but components failed for other reasons
        st.warning(f"⚠️ RAG components could not be loaded correctly. Some features might be unavailable. Status: {st.session_state.get('component_status_message', 'Unknown error')}", icon="🛠️")
elif not project_modules_loaded:
    st.error("🚨 **Critical Error:** Core project modules could not be loaded. The application cannot function. Please check the logs.", icon="❌")


# --- Main Application UI ---
# Initialize session state variables for UI if they don't exist
if 'recommendation_processing' not in st.session_state:
    st.session_state.recommendation_processing = False
if 'llm_answer' not in st.session_state:
    st.session_state.llm_answer = None
if 'context_sources' not in st.session_state:
    st.session_state.context_sources = []


# --- UI Tabs (with Icons) ---
tab_rec, tab_how, tab_about, tab_fetch, tab_arxiv, tab_feedback = st.tabs([
    "🧠 **Recommend**",
    "⚙️ How It Works",
    "ℹ️ About",
    "⏬ Update Knowledge Base",
    "🔍 Search arXiv",
    "📝 Feedback"
])

# --- Recommendation Tab ---
with tab_rec:
    st.header("💬 Ask the RAG Assistant")
    st.markdown("Enter your research topic or question below. The assistant will retrieve relevant information from the indexed documents and generate an answer.")

    col1_rec, col2_rec = st.columns([3, 1])

    with col1_rec:
        default_query_val = "Example: Explain Retrieval-Augmented Generation (RAG)."
        if project_modules_loaded and config and hasattr(config, 'DEFAULT_QUERY'):
            default_query_val = config.DEFAULT_QUERY
        query = st.text_area(
            "Your Question:",
            value=st.session_state.get("rec_query", default_query_val),
            height=150,
            key="rec_query_input_area",
            placeholder="Type your question about the research papers here...",
            help="Ask anything related to the content of the indexed documents.",
            label_visibility="collapsed"
        )

    with col2_rec:
        st.markdown("**Options**")
        general_mode = st.toggle(
            "Hybrid Mode",
            value=st.session_state.get("rec_general", False),
            key="rec_general_toggle",
            help="Allows the AI to use its general knowledge *in addition* to the retrieved documents. 'Strict Mode' (off) uses *only* document context."
        )
        concise_mode = st.toggle(
            "Concise Prompt",
            value=st.session_state.get("rec_concise", True),
            key="rec_concise_toggle",
            help="Uses a more structured, concise prompt for the AI, potentially faster but less conversational."
        )
        st.markdown("---")
        submit_rec = st.button(
            "✨ Get Recommendation",
            type="primary",
            key="rec_button",
            disabled=not query or not project_modules_loaded,
            use_container_width=True
        )
        
        _components_ok_rec_tab = st.session_state.get('components_loaded_successfully', False)
        _kb_empty_rec_tab = st.session_state.get('knowledge_base_is_empty', True)

        if not _components_ok_rec_tab:
            st.caption("⚠️ **RAG components are not fully loaded.** Recommendations might be limited or unavailable. Check setup or refresh.") # Icon removed
        elif _kb_empty_rec_tab:
            st.caption("💡 **Knowledge base is empty.** The AI will use its general knowledge. Add papers in 'Update Knowledge Base' for document-specific answers.") # Icon removed


    if submit_rec:
        if not project_modules_loaded or not cli_run_recommendation_async:
            st.error("🚨 **Critical Error:** Recommendation system is not available due to missing core modules.", icon="❌")
            logger.error("Submit_rec: Attempted to run recommendation but project_modules_loaded is False or cli_run_recommendation_async is None.")
        else:
            st.session_state.rec_query = query
            st.session_state.rec_general = general_mode 
            st.session_state.rec_concise = concise_mode

            _components_loaded_successfully = st.session_state.get('components_loaded_successfully', False)
            _kb_is_empty = st.session_state.get('knowledge_base_is_empty', True)
            
            knowledge_base_effectively_empty = False
            if not _components_loaded_successfully or _kb_is_empty:
                warning_message = "⚠️ **Knowledge base appears to be empty or RAG components are not fully loaded!** The AI will attempt to use its general knowledge only."
                st.warning(warning_message, icon="📚")
                st.markdown("💡 **Recommendation**: For better results with document context, consider adding research papers in the 'Update Knowledge Base' tab or ensuring data files are correctly deployed and components load.")
                logger.warning(f"User attempting recommendation. Query: '{query[:50]}...'. Components OK: {_components_loaded_successfully}, KB Empty: {_kb_is_empty}. Forcing Hybrid mode.")
                knowledge_base_effectively_empty = True

            effective_general_mode = general_mode 
            if knowledge_base_effectively_empty:
                effective_general_mode = True 

            mode_name = "Hybrid" if effective_general_mode else "Strict"
            style_name = "Concise" if concise_mode else "Detailed"
            
            info_message = f"Running recommendation with '{mode_name}' RAG and '{style_name}' Prompt..."
            log_message_detail = ""

            if knowledge_base_effectively_empty:
                forced_reason_parts = []
                if not _components_loaded_successfully:
                    forced_reason_parts.append("components not loaded")
                if _kb_is_empty:
                    forced_reason_parts.append("empty KB")
                forced_reason = " and ".join(forced_reason_parts) if forced_reason_parts else "an issue"
                
                info_message += f" (Note: {mode_name} mode forced due to {forced_reason})"
                log_message_detail = f" (forced due to {forced_reason})"
            
            st.info(info_message, icon="⏳")
            logger.info(f"Running recommendation for query: '{query[:50]}...' with Mode='{mode_name}'{log_message_detail}, Style='{style_name}'")

            st.session_state.llm_answer = None 
            st.session_state.context_sources = []
            st.session_state.raw_context_chunks = [] 
            answer_placeholder = st.empty() 

            top_n_val = 5 
            if config: 
                try:
                    top_n_val = config.TOP_N_RESULTS
                except AttributeError as e:
                    logger.warning(f"Could not get config.TOP_N_RESULTS ({e}), using default {top_n_val}.")
            else:
                logger.warning("Config object not available for TOP_N_RESULTS, using default.")

            try:
                asyncio.run(handle_recommendation_submission_async(
                    query_text=query,
                    top_n_results=top_n_val,
                    effective_general_mode=effective_general_mode,
                    use_concise_mode=concise_mode,
                    answer_placeholder=answer_placeholder
                ))
            except Exception as e:
                st.error("🤖 **Oops! A critical error occurred while trying to start the recommendation.** Please refresh and try again.", icon="❌")
                logger.error(f"Critical error in submit_rec block (asyncio.run wrapper): {e}", exc_info=True)
                if answer_placeholder: answer_placeholder.empty() 

    # --- Display Results Section ---
    if st.session_state.get("llm_answer") is not None:
        st.markdown("---")
        # Answer is primarily displayed by the answer_placeholder in handle_recommendation_submission_async.
        # This st.markdown is a fallback or ensures it is displayed correctly after rerun.
        st.markdown(st.session_state.llm_answer, unsafe_allow_html=True)

        # --- Displaying Retrieved Context ---
        if not st.session_state.recommendation_processing and st.session_state.llm_answer:
            if st.session_state.raw_context_chunks: # Changed from context_sources
                with st.expander(f"📚 Retrieved Context ({len(st.session_state.raw_context_chunks)})", expanded=False): # Changed from context_sources
                    # Add debug info if no content is found
                    content_found = False
                    for i, source_item in enumerate(st.session_state.raw_context_chunks): # Changed from context_sources
                        st.markdown(f"**Source {i+1}:**") # Changed to i+1 for 1-based indexing
                        content = source_item.get('content', source_item.get('chunk_text', 'N/A'))
                        if content != 'N/A':
                            content_found = True
                        metadata = source_item.get('metadata', {})
                        title = metadata.get('title', source_item.get('original_title', 'Unknown Title'))
                        page_num = metadata.get('page_number', 'N/A') 
                        url = metadata.get('url', source_item.get('original_url', '#'))

                        st.text(content[:500] + "..." if len(content) > 500 else content)
                        st.caption(f"📄 {title} | Page: {page_num} | URL: {url}")
                        st.markdown("---")
                        
                    # Show debug information if needed
                    if not content_found and st.session_state.raw_context_chunks:
                        with st.expander("🔍 Debug Information", expanded=False):
                            st.write("No content was found in the retrieved chunks. Here are the available keys:")
                            sample_chunk = st.session_state.raw_context_chunks[0]
                            st.json(list(sample_chunk.keys()))
        
        if st.button("🗑️ Clear Results", key="clear_rec_results_button_main_display"):
            st.session_state.llm_answer = None
            st.session_state.context_sources = []
            st.session_state.raw_context_chunks = []
            st.rerun()

# --- How It Works Tab ---
with tab_how:
    st.header("⚙️ How the RAG System Works")
    # ... (content is mostly static, ensure config references are safe) ...
    emb_model_display = "a sentence transformer model"
    if project_modules_loaded and config and hasattr(config, 'EMBEDDING_MODEL_NAME'):
        emb_model_display = f"`{config.EMBEDDING_MODEL_NAME}`"
    
    llm_provider_display = "a Large Language Model (LLM)"
    if project_modules_loaded and config and hasattr(config, 'LLM_PROVIDER_ORDER') and config.LLM_PROVIDER_ORDER:
        primary_provider = config.LLM_PROVIDER_ORDER[0].capitalize()
        fallback_providers = [p.capitalize() for p in config.LLM_PROVIDER_ORDER[1:]]
        llm_provider_display = primary_provider
        if fallback_providers:
            llm_provider_display += f" (with fallbacks: {', '.join(fallback_providers)})"

    st.markdown(f"""
    * **Vector Embeddings:** Creates numerical representations (vectors) capturing semantic meaning using {emb_model_display}.
    * **Keyword Index (BM25):** Creates a traditional keyword index for term matches.
    * **Combined Power:** Stores both index types locally.
    """)
    st.markdown(f"""
    * **Context Injection:** Top-ranked retrieved chunks are passed as context to {llm_provider_display}.
    * **Informed Answering:** The LLM generates an answer based on the provided context (Strict RAG) or a mix (Hybrid Mode).
    """)
    # ... (rest of How It Works content) ...

# --- About Tab ---
with tab_about:
    st.header("ℹ️ About This Project")
    # ... (content is static) ...
    st.link_button("View Project on GitHub", "https://github.com/Fane-Nathan/Research-Assistant/tree/alpha-release")


# --- Fetch Data Tab ---
with tab_fetch:
    st.header("⏬ Update Knowledge Base")
    st.markdown("Fetch new data from arXiv or web URLs to update the local knowledge base. This involves fetching, chunking, embedding, and indexing.")

    # Initialize session state for fetch tab inputs if not present
    if 'fetch_arxiv_query' not in st.session_state: st.session_state.fetch_arxiv_query = "large language models"
    if 'fetch_num_arxiv' not in st.session_state: st.session_state.fetch_num_arxiv = 10
    # ... (other fetch tab initializations) ...
    if 'fetch_web_url' not in st.session_state: st.session_state.fetch_web_url = ""


    use_suggestions = st.checkbox(
        "Suggest sources using LLM (overrides arXiv query below if successful)",
        value=st.session_state.get("fetch_suggest_sources", False), # Use .get for safety
        key="fetch_suggest_checkbox"
    )
    st.session_state.fetch_suggest_sources = use_suggestions # Update session state

    if use_suggestions:
        # ... (UI for suggestions) ...
        pass
    else:
        arxiv_query_fetch = st.text_input(
            "arXiv Query:", value=st.session_state.fetch_arxiv_query, key="fetch_arxiv_query_input"
        )
        st.session_state.fetch_arxiv_query = arxiv_query_fetch
        num_arxiv_fetch = st.number_input(
            "Max arXiv Results (0 to skip arXiv):", min_value=0, value=st.session_state.fetch_num_arxiv, step=1, key="fetch_num_arxiv_input"
        )
        st.session_state.fetch_num_arxiv = num_arxiv_fetch


    web_url_fetch_val = st.text_input(
        "Single Web URL to Crawl (Optional):",
        value=st.session_state.get("fetch_web_url", ""),
        key="fetch_web_url_input",
        placeholder="https://example.com/article"
    )
    st.session_state.fetch_web_url = web_url_fetch_val


    fetch_data_button = st.button("🚀 Fetch and Process Data", type="primary", disabled=not project_modules_loaded)
    
    if not project_modules_loaded:
        st.caption("🔧 **System initialization required** - Please refresh the page to enable data fetching.")

    if fetch_data_button and project_modules_loaded and setup_data_and_fetch_cli and config:
        with st.spinner("Fetching and processing data... This may take a while."):
            try:
                # Prepare arguments for setup_data_and_fetch_cli
                # This part needs to correctly construct the 'args' namespace object
                # that setup_data_and_fetch_cli expects.
                
                # Create a mock argparse.Namespace object
                fetch_cli_args = argparse.Namespace()
                fetch_cli_args.suggest_sources = st.session_state.get("fetch_suggest_sources", False)
                fetch_cli_args.topic = st.session_state.get("fetch_topic_for_suggestion", "") if fetch_cli_args.suggest_sources else ""
                
                # Handle arXiv query and num_arxiv based on whether suggestions are used
                if fetch_cli_args.suggest_sources:
                    # If suggesting, num_arxiv might still be relevant if suggestions include arXiv queries
                    fetch_cli_args.arxiv_query = "" # Let suggestions fill this if any
                    fetch_cli_args.num_arxiv = st.session_state.get("fetch_num_arxiv", 10) # Or a default for suggested queries
                else:
                    fetch_cli_args.arxiv_query = st.session_state.get("fetch_arxiv_query", "")
                    fetch_cli_args.num_arxiv = st.session_state.get("fetch_num_arxiv", 0) # If no query, num_arxiv should be 0

                # Handle web URLs
                # setup_data_and_fetch_cli might expect TARGET_WEB_URLS in config to be modified
                # or accept URLs directly. For simplicity, let's assume it uses config.TARGET_WEB_URLS
                # and we modify it if a single URL is provided.
                original_target_urls = list(config.TARGET_WEB_URLS) # Backup
                single_web_url = st.session_state.get("fetch_web_url", "").strip()
                if single_web_url:
                    config.TARGET_WEB_URLS = [single_web_url] + original_target_urls # Prepend
                    # If only single URL and no arXiv, ensure arXiv params are non-interfering
                    if not fetch_cli_args.arxiv_query and fetch_cli_args.num_arxiv == 0 and not fetch_cli_args.suggest_sources:
                         fetch_cli_args.arxiv_query = "dummy_query_for_parser_if_needed" # Placeholder if parser requires it
                         fetch_cli_args.num_arxiv = 0
                
                # num_web_pages argument
                fetch_cli_args.num_web_pages = config.MAX_PAGES_TO_CRAWL # Default, can be overridden if UI provides it

                status_message = asyncio.run(setup_data_and_fetch_cli(fetch_cli_args))
                
                # Restore original config.TARGET_WEB_URLS if modified
                if single_web_url:
                    config.TARGET_WEB_URLS = original_target_urls

                st.success(status_message) # Display status from CLI function
                logger.info(f"Data fetch status from CLI: {status_message}")

                # Crucially, re-initialize components to load the new data
                st.session_state.force_component_reload = True
                initialize_and_load_components()
                
                if st.session_state.get('components_loaded_successfully', False):
                    st.toast("RAG Components reloaded with new data.", icon="🔄")
                    if st.session_state.get('knowledge_base_is_empty', True):
                         st.warning("Data fetched, but the knowledge base still appears empty or components didn't fully utilize it. Please check logs.", icon="⚠️")
                    else:
                         st.success("Knowledge base updated and RAG components are ready!", icon="✅")
                else:
                    st.warning("🔄 **Components couldn't reload automatically after data fetch.** Your data might have been fetched, but you may need to refresh the page to use it in the Recommend tab.", icon="⚠️")
                
                st.rerun() # Rerun to update UI elements based on new state

            except ImportError:
                st.error("🔧 **Technical issue detected.** The data fetching system couldn't load properly. Please refresh the page or contact support if the problem persists.", icon="❌")
                logger.error("ImportError for setup_data_and_fetch_cli in Streamlit app.")
            except Exception as e:
                st.error(f"📡 **Data fetch encountered an issue:** {e}. This might be temporary - please try again. If it persists, check logs or your inputs.", icon="🌐")
                logger.error(f"Data fetch error in Streamlit UI: {e}", exc_info=True)
    elif fetch_data_button and (not project_modules_loaded or not setup_data_and_fetch_cli or not config) :
         st.error("🚨 **Critical Error:** Data fetching system is not available due to missing core modules. Please check the application setup.", icon="❌")


# --- Search arXiv Tab ---
with tab_arxiv:
    st.header("🔍 Search arXiv Directly")
    # ... (content is mostly fine, ensure run_arxiv_search_cli is used) ...
    if 'arxiv_query_direct' not in st.session_state: st.session_state.arxiv_query_direct = "quantum machine learning"
    # ...
    submit_arxiv_search = st.button("Search arXiv", type="primary", disabled=not project_modules_loaded or not st.session_state.get("arxiv_query_direct", ""))

    if submit_arxiv_search and project_modules_loaded and run_arxiv_search_cli:
        # ... (logic to call run_arxiv_search_cli and display results) ...
        logger.info(f"Performing direct arXiv search for: '{st.session_state.arxiv_query_direct}', num_results={st.session_state.get('arxiv_num_results_direct', 5)}")
        with st.spinner("Searching arXiv..."):
            try:
                results = run_arxiv_search_cli(st.session_state.arxiv_query_direct, st.session_state.get('arxiv_num_results_direct', 5))
                if results:
                    # ... (display logic for results) ...
                    pass
                else:
                    st.info("No results found for your query on arXiv.")
            except Exception as e:
                st.error(f"🔍 **arXiv search encountered an issue:** {e}", icon="🌐")
                logger.error(f"Direct arXiv search error in UI: {e}", exc_info=True)
    elif submit_arxiv_search and (not project_modules_loaded or not run_arxiv_search_cli):
        st.error("🚨 **Critical Error:** arXiv search functionality is not available due to missing core modules.", icon="❌")


# --- Feedback Tab ---
with tab_feedback:
    st.header("📝 Submit Feedback or Report Issues")
    # ... (content is static) ...
    try:
        components.iframe("https://tally.so/embed/n0kkp6?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1", height=500, scrolling=True)
    except Exception as e:
        # ... (fallback link) ...
        pass

# --- Footer ---
st.divider()

