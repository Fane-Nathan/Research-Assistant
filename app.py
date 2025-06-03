# -*- coding: utf-8 -*-
"""
Streamlit web application for the Study Assistant (User Provided Version - Improved).

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
import types # For checking generator types
import argparse # For creating namespace objects for CLI functions
import streamlit.components.v1 as components
import re # For text processing (e.g., URL parsing in context display)

from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional, Generator # Ensure Any and Generator are imported
import numpy as np # Keep if used by components

# Import text cleaning functions for academic content
from hybrid_search_rag.text_processing.text_cleaner import (
    clean_academic_title, clean_academic_content, clean_context_list, 
    clean_academic_text
)

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [App-Console] %(message)s',
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Ensure log file path is robust, especially in different deployment environments
log_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True) # Create log directory if it doesn't exist
log_file_path = os.path.join(log_dir, "app.log")

try:
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    st.warning(f"Could not set up file logging: {e}") # Non-critical, app can still run
    logger.error(f"Failed to set up file handler for logging: {e}")


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
check_nltk_data_cli = None
load_components_cli = None
setup_data_and_fetch_cli = None
run_arxiv_search_cli = None
cli_run_recommendation_async = None

try:
    from hybrid_search_rag import config
    from scripts.cli import (
        load_components as load_components_cli,
        setup_data_and_fetch as setup_data_and_fetch_cli,
        run_arxiv_search as run_arxiv_search_cli,
        run_recommendation as cli_run_recommendation_async,
        check_nltk_data as check_nltk_data_cli,
        chunk_text_by_sentences 
    )
    project_modules_loaded = True
    logger.info("Project module imports successful.")
except ImportError as e:
    st.error("🔧 **System setup incomplete.** Please ensure all required components are properly installed and refresh the page. If the issue persists, check the installation guide or contact support.", icon="📋")
    st.code(f"Current sys.path: {sys.path}\nImport Error: {e}")
    logger.error(f"Project import failed (ImportError): {e}", exc_info=True)
except Exception as e:
    st.error("🔧 **Startup issue detected.** Some features may not work correctly. Please refresh the page or contact support if the problem persists.", icon="⚠️")
    logger.error(f"Project import failed (Exception): {e}", exc_info=True)

# --- Windows asyncio policy fix ---
if sys.platform == "win32" and "asyncio" in sys.modules:
    logger.info("Applying WindowsProactorEventLoopPolicy for asyncio.")
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- Inject Custom CSS for Context Cards ---
st.markdown("""
<style>
.context-card {
  border: 1px solid #e0e0e0;
  border-radius: 8px; /* Slightly more rounded */
  padding: 15px;
  margin-bottom: 15px;
  background-color: #f9f9f9; /* Light background for the card */
  box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Subtle shadow */
}
.context-card h5 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #333; /* Darker title color */
}
.context-card small {
  display: block; /* Ensure small tags take full width if needed */
  line-height: 1.5;
  color: #555; /* Slightly lighter text for abstract/snippet */
}
.metadata-line {
  font-size: 0.85em; /* Slightly smaller metadata */
  color: #666; /* Grey color for metadata */
  margin-top: 10px;
}
.metadata-line a {
    color: #007bff; /* Standard link color */
    text-decoration: none;
}
.metadata-line a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# --- Centralized Component Initialization and State Management ---
def initialize_and_load_components():
    """
    Initializes and loads RAG components, updating session state.
    This function is designed to be called on app start and after data updates.
    """
    logger.info("Attempting to initialize and load RAG components...")
    if not project_modules_loaded or not load_components_cli:
        logger.error("Cannot initialize: project modules or load_components_cli not available.")
        st.session_state.components_loaded_successfully = False
        st.session_state.rag_components = None
        st.session_state.knowledge_base_is_empty = True # Default assumption
        st.session_state.component_status_message = "Error: Core application modules not loaded."
        return

    if check_nltk_data_cli and not check_nltk_data_cli():
        logger.warning("NLTK data check failed prior to loading components. Proceeding with load attempt.")
        # load_components_cli should ideally also log/handle this if it's critical for its operation

    force_reload = st.session_state.get("force_component_reload", False)
    logger.info(f"Calling load_components_cli with force_reload={force_reload}")
    
    # EXPECTATION from load_components_cli:
    # Returns a dictionary that should ideally always contain:
    # 'recommender': (object or None), 
    # 'data_manager': (object or None), # Or similar for KB status check
    # 'knowledge_base_is_empty': bool,  # Crucial for UI state
    # 'status_message': str             # Informative message
    loaded_data_dict = load_components_cli(force_reload=force_reload)
    st.session_state.force_component_reload = False 

    if loaded_data_dict:
        st.session_state.rag_components = loaded_data_dict
        st.session_state.components_loaded_successfully = bool(loaded_data_dict.get("recommender"))
        
        if "knowledge_base_is_empty" in loaded_data_dict:
            st.session_state.knowledge_base_is_empty = loaded_data_dict["knowledge_base_is_empty"]
        elif loaded_data_dict.get("data_manager") and hasattr(loaded_data_dict["data_manager"], 'is_data_empty'):
            st.session_state.knowledge_base_is_empty = loaded_data_dict["data_manager"].is_data_empty()
            logger.info("Used data_manager.is_data_empty() for KB status.")
        else:
            logger.warning("knowledge_base_is_empty not directly provided by load_components_cli, nor via data_manager.is_data_empty. Performing fallback file check.")
            kb_files_found_fallback = False
            if config and config.DATA_DIR and config.METADATA_FILE:
                metadata_file_path = os.path.join(config.DATA_DIR, config.METADATA_FILE)
                if os.path.exists(metadata_file_path) and os.path.getsize(metadata_file_path) > 10:
                    try:
                        with open(metadata_file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                        if isinstance(content, list) and len(content) > 0:
                            kb_files_found_fallback = True
                    except Exception as e_fallback:
                        logger.error(f"Error during fallback KB metadata check: {e_fallback}")
            st.session_state.knowledge_base_is_empty = not kb_files_found_fallback
            logger.info(f"Fallback KB check result: empty = {st.session_state.knowledge_base_is_empty}")

        st.session_state.component_status_message = loaded_data_dict.get("status_message", "Components status unknown.")
        
        if st.session_state.components_loaded_successfully:
            logger.info(f"Components initialized successfully. KB empty: {st.session_state.knowledge_base_is_empty}. Status: {st.session_state.component_status_message}")
        else:
            logger.warning(f"Component initialization failed or recommender not loaded. KB empty: {st.session_state.knowledge_base_is_empty}. Status: {st.session_state.component_status_message}")
    else:
        logger.error("load_components_cli returned None or an empty dictionary. Critical failure.")
        st.session_state.components_loaded_successfully = False
        st.session_state.rag_components = None
        st.session_state.knowledge_base_is_empty = True
        st.session_state.component_status_message = "Critical error: Component loader failed to return status."


async def handle_recommendation_submission_async(
    query_text: str,
    top_n_results: int,
    effective_general_mode: bool,
    use_concise_mode: bool,
    answer_placeholder: Any,
):
    """
    Handles the asynchronous submission of a recommendation request.
    """
    st.session_state.recommendation_processing = True
    st.session_state.llm_answer = ""
    st.session_state.context_sources = []
    st.session_state.raw_context_chunks = []
    current_answer_display = ""

    try:
        logger.info(f"handle_recommendation_submission_async: Calling cli_run_recommendation_async for query '{query_text[:50]}...'")
        
        if not cli_run_recommendation_async: 
            logger.error("cli_run_recommendation_async is not available!")
            st.session_state.llm_answer = "Error: Recommendation function not loaded."
            answer_placeholder.error(st.session_state.llm_answer)
            return

        # This assumes cli_run_recommendation_async is an awaitable that returns a tuple:
        # (answer_data, sources_data, raw_chunks_data)
        # where answer_data could be a string or a synchronous generator.
        result = await cli_run_recommendation_async(
            query=query_text,
            num_final_results=top_n_results,
            general_mode=effective_general_mode,
            concise_mode=use_concise_mode,
            # rag_components are accessed by cli_run_recommendation_async internally from st.session_state
        )

        if result:
            answer_data, sources_data, raw_chunks_data = result

            st.session_state.context_sources = sources_data if sources_data else []
            st.session_state.raw_context_chunks = raw_chunks_data if raw_chunks_data else []

            if isinstance(answer_data, types.GeneratorType): 
                for token in answer_data:
                    current_answer_display += token
                    answer_placeholder.markdown(current_answer_display + "▌") 
                st.session_state.llm_answer = current_answer_display
                if current_answer_display:
                    answer_placeholder.markdown(current_answer_display)
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
            # If cli_run_recommendation_async were to return an AsyncGenerator:
            # elif isinstance(answer_data, types.AsyncGeneratorType):
            #     async for token in answer_data:
            #         current_answer_display += token
            #         answer_placeholder.markdown(current_answer_display + "▌")
            #     st.session_state.llm_answer = current_answer_display
            #     if current_answer_display:
            #         answer_placeholder.markdown(current_answer_display)
            #     else: # ...
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
            check_nltk_data_cli()
            st.session_state.nltk_data_checked_app = True
            logger.info("APP.PY: Initial NLTK check complete.")
    elif not project_modules_loaded:
        logger.warning("APP.PY: NLTK check skipped: project_modules_loaded is False.")
    else:
        logger.warning("APP.PY: NLTK check skipped: check_nltk_data_cli function not available.")
        st.warning("🔧 **Setup Incomplete:** Core language components (NLTK check function) are not available. Please wait or refresh.", icon="⏳")


# --- Initial Component Load (once per session or after forced reload) ---
if 'components_loaded_successfully' not in st.session_state or st.session_state.get("force_component_reload"):
    st.session_state.force_component_reload = True 
    initialize_and_load_components()

# Log current status after potential initialization
logger.info(f"App Rerun. Modules loaded: {project_modules_loaded}, Components loaded: {st.session_state.get('components_loaded_successfully', False)}, KB empty: {st.session_state.get('knowledge_base_is_empty', True)}")
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
if not project_modules_loaded:
    st.error("🚨 **Critical Error:** Core project modules could not be loaded. The application cannot function. Please check the logs and setup.", icon="❌")
elif not st.session_state.get('components_loaded_successfully', False):
    st.warning(f"⚠️ RAG components are not fully loaded. Some features might be unavailable. Status: {st.session_state.get('component_status_message', 'Unknown error')}. Try refreshing or checking logs.", icon="🛠️")
    if st.session_state.get('knowledge_base_is_empty', True):
         st.info("🎯 **Welcome!** To get started, add research papers via the **'Update Knowledge Base'** tab. This enables AI recommendations based on your documents.")
elif st.session_state.get('knowledge_base_is_empty', True):
    st.info("🎯 **Welcome to Research Assistant!** Your knowledge base is currently empty. Please add some research papers using the **'Update Knowledge Base'** tab below to enable document-specific AI recommendations.")
    st.markdown("💡 **Tip:** You can search arXiv for topics like 'machine learning', 'quantum computing', or add specific web URLs.")
# else: All systems go, KB has data. No specific welcome message needed here.


# --- Main Application UI ---
# Initialize session state variables for UI if they don't exist
if 'recommendation_processing' not in st.session_state: st.session_state.recommendation_processing = False
if 'llm_answer' not in st.session_state: st.session_state.llm_answer = None
if 'context_sources' not in st.session_state: st.session_state.context_sources = []
if 'raw_context_chunks' not in st.session_state: st.session_state.raw_context_chunks = []


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
    st.markdown("Enter your research topic or question. The assistant will use indexed documents and AI to generate an answer.")

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

    # Determine overall system readiness for RAG in this tab
    _modules_ok_rec_tab = project_modules_loaded and cli_run_recommendation_async is not None
    _components_ok_rec_tab = st.session_state.get('components_loaded_successfully', False)
    _kb_ready_rec_tab = not st.session_state.get('knowledge_base_is_empty', True)
    
    is_rag_fully_functional_rec_tab = _modules_ok_rec_tab and _components_ok_rec_tab
    can_use_kb_rec_tab = is_rag_fully_functional_rec_tab and _kb_ready_rec_tab

    with col2_rec:
        st.markdown("**Options**")
        general_mode = st.toggle(
            "Hybrid Mode",
            value=st.session_state.get("rec_general", False),
            key="rec_general_toggle",
            help="Allows AI to use its general knowledge *in addition* to retrieved documents. 'Strict Mode' (off) uses *only* document context."
        )
        concise_mode = st.toggle(
            "Concise Prompt",
            value=st.session_state.get("rec_concise", True),
            key="rec_concise_toggle",
            help="Uses a more structured, concise prompt for the AI, potentially faster but less conversational."
        )
        st.markdown("---")
        
        submit_rec_disabled = not query or not _modules_ok_rec_tab
        submit_rec_tooltip = ""
        if not query:
            submit_rec_tooltip = "Please enter a question."
        elif not _modules_ok_rec_tab:
            submit_rec_tooltip = "Recommendation system is not available (core modules missing)."
        elif not _components_ok_rec_tab:
             submit_rec_tooltip = "RAG components are not fully loaded. Recommendations might be limited."
        
        submit_rec = st.button(
            "✨ Get Recommendation",
            type="primary",
            key="rec_button",
            disabled=submit_rec_disabled,
            help=submit_rec_tooltip if submit_rec_tooltip else "Submit your question to the RAG assistant.",
            use_container_width=True
        )
        
        # Captions for status in options column
        if not _modules_ok_rec_tab:
            st.caption("🚫 **Critical:** Core recommendation modules missing.")
        elif not _components_ok_rec_tab:
            st.caption(f"⚠️ **RAG components issue:** {st.session_state.get('component_status_message', 'Not fully loaded')}. AI might use general knowledge only.")
        elif not _kb_ready_rec_tab: 
            st.caption("💡 **Knowledge base empty.** AI will use general knowledge. Add papers via 'Update KB' tab.")
        # else: All systems go for RAG.

    if submit_rec: 
        st.session_state.rec_query = query
        st.session_state.rec_general = general_mode 
        st.session_state.rec_concise = concise_mode

        effective_general_mode = general_mode
        knowledge_base_issue_message = ""

        if not _components_ok_rec_tab:
            knowledge_base_issue_message = "RAG components not fully loaded. Forcing Hybrid mode (using general AI knowledge)."
            effective_general_mode = True
        elif not _kb_ready_rec_tab:
            knowledge_base_issue_message = "Knowledge base is empty. Forcing Hybrid mode (using general AI knowledge)."
            effective_general_mode = True 
        if knowledge_base_issue_message:
            st.warning(f"⚠️ {knowledge_base_issue_message}", icon="📚")
            logger.warning(f"User query '{query[:50]}...'. {knowledge_base_issue_message}")

        mode_name = "Hybrid" if effective_general_mode else "Strict"
        style_name = "Concise" if concise_mode else "Detailed"
        
        info_msg_parts = [f"Running recommendation with '{mode_name}' RAG and '{style_name}' Prompt..."]
        log_msg_detail_parts = []

        if effective_general_mode and not general_mode:
            reason = "RAG components not loaded" if not _components_ok_rec_tab else "empty Knowledge Base"
            forced_info = f"(Hybrid mode was automatically enabled because the {reason})."
            info_msg_parts.append(forced_info)
            log_msg_detail_parts.append(f"(forced due to {reason})")
        
        st.info(" ".join(info_msg_parts), icon="⏳")
        logger.info(f"Running recommendation for query: '{query[:50]}...' Mode='{mode_name}' {''.join(log_msg_detail_parts)}, Style='{style_name}'")

        st.session_state.llm_answer = None 
        st.session_state.context_sources = []
        st.session_state.raw_context_chunks = [] 
        answer_placeholder = st.empty() 

        top_n_val = 5 
        if config and hasattr(config, 'TOP_N_RESULTS'): 
            try:
                top_n_val = int(config.TOP_N_RESULTS)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse config.TOP_N_RESULTS ({config.TOP_N_RESULTS}) as int ({e}), using default {top_n_val}.")
        else:
            logger.warning(f"Config object or TOP_N_RESULTS not available, using default top_n_val: {top_n_val}.")

        try:
            asyncio.run(handle_recommendation_submission_async(
                query_text=query,
                top_n_results=top_n_val,
                effective_general_mode=effective_general_mode,
                use_concise_mode=concise_mode,
                answer_placeholder=answer_placeholder
            ))
        except Exception as e: # Catch errors from asyncio.run or the async function itself
            st.error("🤖 **Oops! A critical error occurred while trying to start the recommendation.** Please refresh and try again.", icon="❌")
            logger.error(f"Critical error in submit_rec block (asyncio.run wrapper or async func): {e}", exc_info=True)
            if answer_placeholder: answer_placeholder.empty()

    # --- Display Results Section ---
    if st.session_state.get("llm_answer") is not None: 
        st.markdown("---")
        # The answer_placeholder in handle_recommendation_submission_async handles live updates.
        # This st.markdown is a fallback or ensures it's displayed correctly after rerun if placeholder was cleared.
        if not st.session_state.recommendation_processing: # Only display if not actively processing
             st.markdown(st.session_state.llm_answer, unsafe_allow_html=True)

        # --- Displaying Retrieved Context ---
        if not st.session_state.recommendation_processing and st.session_state.llm_answer:
            if st.session_state.raw_context_chunks:
                with st.expander(f"📚 Retrieved Context ({len(st.session_state.raw_context_chunks)} sources)", expanded=False):
                    cleaned_chunks = clean_context_list(st.session_state.raw_context_chunks)
                    
                    content_found_in_chunks = False
                    context_message_placeholder = st.empty()
                    
                    for i, source_item in enumerate(cleaned_chunks):
                        title = source_item.get('title', "No Title Available")
                        abstract = source_item.get('abstract')
                        authors = source_item.get('authors')
                        organization = source_item.get('metadata', {}).get('organization', source_item.get('metadata', {}).get('institution'))
                        url = source_item.get('url')
                        page_num = source_item.get('page_number')
                        
                        # Determine source type more robustly
                        source_type = source_item.get('source_type', source_item.get('metadata', {}).get('type'))

                        if isinstance(url, str) and url.startswith('arxiv:'):
                            arxiv_id = url.split(':')[-1]
                            url = f"https://arxiv.org/abs/{arxiv_id}"
                        
                        with st.container(): 
                            st.markdown(f'<div class="context-card">', unsafe_allow_html=True)
                            
                            has_card_content = False
                            if title and title != "No Title Available":
                                st.markdown(f"##### 📄 {title}")
                                has_card_content = True
                            
                            if abstract and abstract.strip() and abstract.lower() not in ["content not available. this may be a reference-only entry.", "n/a"]:
                                st.markdown("**Abstract:**")
                                st.markdown(f"<small>{abstract}</small>", unsafe_allow_html=True)
                                has_card_content = True
                            elif source_item.get('content') and source_item.get('content').strip().lower() not in ["content not available. this may be a reference-only entry.", "no content available for this source.", "n/a"]:
                                content_snippet = source_item['content']
                                snippet_display = (content_snippet[:280] + '...') if len(content_snippet) > 280 else content_snippet
                                st.markdown("**Content Snippet:**")
                                st.markdown(f"<small>{snippet_display}</small>", unsafe_allow_html=True)
                                has_card_content = True
                            
                            if not has_card_content and (not title or title == "No Title Available"): 
                                st.markdown("<small><em>Content/Abstract not available for this item. Displaying available metadata.</em></small>", unsafe_allow_html=True)


                            if authors:
                                author_str = ', '.join(authors) if isinstance(authors, list) else str(authors)
                                st.markdown(f"**Authors:** {author_str}")
                            if organization:
                                st.markdown(f"**Organization:** {organization}")

                            metadata_parts = []
                            if url and url not in ["#", "N/A", ""]:
                                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                                display_url_text = domain_match.group(1) if domain_match else "Source Link"
                                metadata_parts.append(f"**Source:** [{display_url_text}]({url})")
                            if page_num and str(page_num).lower() not in ["n/a", ""]:
                                metadata_parts.append(f"**Page:** {page_num}")
                            if source_type and str(source_type).strip().lower() not in ["n/a", "", "unknown source", "unknown", "none"]:
                                metadata_parts.append(f"**Type:** {source_type.capitalize()}")
                            
                            if metadata_parts:
                                st.markdown(f'<div class="metadata-line">{" | ".join(metadata_parts)}</div>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            if i < len(cleaned_chunks) -1 : st.markdown("---")

                            if has_card_content: content_found_in_chunks = True
                    
                    if not content_found_in_chunks and cleaned_chunks:
                        context_message_placeholder.warning("No primary content (titles, abstracts, or snippets) was found in the retrieved context sources, though metadata might be present. The AI's answer might be less specific.", icon="ℹ️")
                    elif not cleaned_chunks and st.session_state.llm_answer:
                         context_message_placeholder.info("No specific documents were retrieved from the knowledge base for this query. The AI answered from its general knowledge.", icon="💡")


                    # Debugging section if no content found in any chunks (original logic retained and slightly enhanced)
                    if not content_found_in_chunks and st.session_state.raw_context_chunks:
                        with st.expander("🔍 Debug: Retrieved Context Structure (No Displayable Content Found)", expanded=False): # Default to collapsed
                            st.warning("No displayable content (title, abstract, snippet) was found in the retrieved chunks. This could indicate an issue with data extraction, field naming, or the cleaning process. Below is a sample of the raw data for the first chunk:")
                            sample_chunk_debug = st.session_state.raw_context_chunks[0]
                            
                            st.markdown("##### Available Keys in First Raw Chunk:")
                            st.json(list(sample_chunk_debug.keys()))

                            potential_fields_debug = {
                                k: (str(v)[:150] + '...' if isinstance(v, str) and len(str(v)) > 150 else v)
                                for k, v in sample_chunk_debug.items()
                            }
                            st.markdown("##### Sample Data from First Raw Chunk (truncated):")
                            st.json(potential_fields_debug)
                            
                            st.markdown("###### Troubleshooting Suggestions:")
                            st.markdown("- Verify that your data ingestion process correctly extracts text into expected fields (e.g., 'title', 'abstract', 'content').\n"
                                        "- Check `hybrid_search_rag.text_processing.text_cleaner.clean_context_list` to ensure it's looking for the correct fields.\n"
                                        "- Ensure documents in your knowledge base actually contain textual content.")
                            if st.button("🔄 Force Reload Components & Retry", key="debug_reload_components_context"):
                                st.session_state.force_component_reload = True
                                st.rerun()
            elif st.session_state.llm_answer:
                 st.info("The AI provided an answer, but no specific documents were retrieved from the knowledge base to support it. This might happen if the query was general or the knowledge base is empty/irrelevant.", icon="💡")


        if st.button("🗑️ Clear Results", key="clear_rec_results_button_main_display"):
            st.session_state.llm_answer = None
            st.session_state.context_sources = []
            st.session_state.raw_context_chunks = []
            # Potentially clear the query input as well, or keep it for refinement
            # st.session_state.rec_query = "" # Uncomment to clear query too
            st.rerun()

# --- How It Works Tab ---
with tab_how:
    st.header("⚙️ How the RAG System Works")
    st.markdown("""
    This Research Assistant employs a **Retrieval-Augmented Generation (RAG)** architecture
    to provide answers based on a specialized knowledge base of academic papers.
    Here's a simplified overview:
    """)

    st.subheader("1. Knowledge Base Creation (Offline Process)")
    emb_model_display = "a sentence transformer model (e.g., `all-MiniLM-L6-v2`)"
    if project_modules_loaded and config and hasattr(config, 'EMBEDDING_MODEL_NAME') and config.EMBEDDING_MODEL_NAME:
        emb_model_display = f"`{config.EMBEDDING_MODEL_NAME}`"
    
    st.markdown(f"""
    * **Data Ingestion:** Research papers (from arXiv, web URLs) are fetched and parsed.
    * **Text Chunking:** The content of each paper is divided into smaller, manageable chunks.
    * **Embedding Generation:** Each chunk is converted into a numerical vector (embedding) using {emb_model_display}. These embeddings capture the semantic meaning of the text.
    * **Indexing:**
        * **Vector Index:** Embeddings are stored in a specialized database (e.g., FAISS, ChromaDB) for efficient similarity search.
        * **Keyword Index (BM25):** A traditional keyword-based index (like BM25) is also created to catch term-specific matches.
    * **Metadata Storage:** Information like titles, authors, URLs, and source types is stored alongside the content.
    """)

    st.subheader("2. Answering Your Questions (Online Process)")
    llm_provider_display = "a Large Language Model (LLM)"
    if project_modules_loaded and config and hasattr(config, 'LLM_PROVIDER_ORDER') and config.LLM_PROVIDER_ORDER:
        primary_provider = str(config.LLM_PROVIDER_ORDER[0]).capitalize()
        fallback_providers = [str(p).capitalize() for p in config.LLM_PROVIDER_ORDER[1:]]
        llm_provider_display = primary_provider
        if fallback_providers:
            llm_provider_display += f" (with fallbacks like {', '.join(fallback_providers)})"
    
    st.markdown(f"""
    * **Query Input:** You ask a question or provide a research topic.
    * **Hybrid Search & Retrieval:**
        * Your query is also embedded.
        * The system searches both the vector index (for semantic similarity) and the keyword index.
        * Results from both searches are combined and re-ranked to find the most relevant text chunks from the knowledge base.
    * **Context Augmentation:** The top-ranked retrieved chunks are selected as context.
    * **Prompt Engineering:** This context, along with your original query, is formatted into a prompt for {llm_provider_display}.
    * **Answer Generation:**
        * **Strict RAG Mode:** The LLM is instructed to generate an answer *solely* based on the provided context.
        * **Hybrid Mode:** The LLM can use the provided context *and* its general knowledge to formulate a more comprehensive answer.
    * **Response Delivery:** The generated answer is displayed to you, often with links to the source documents.
    """)
    st.markdown("---")
    st.markdown("This RAG approach helps to ground the LLM's responses in factual information from the specified documents, reducing hallucinations and providing more relevant, context-aware answers for research purposes.")

# --- About Tab ---
with tab_about:
    st.header("ℹ️ About This Project")
    st.markdown("""
    This Research Paper Assistant is designed to streamline the process of exploring and understanding academic literature.
    It leverages modern AI techniques, including:

    * **Retrieval-Augmented Generation (RAG):** To provide answers grounded in the content of a curated knowledge base.
    * **Hybrid Search:** Combining semantic vector search with traditional keyword search (BM25) for robust information retrieval.
    * **Large Language Models (LLMs):** For natural language understanding and answer generation.
    * **Streamlit:** For creating this interactive web application.

    **Key Features:**
    * Ask questions in natural language about the indexed research papers.
    * Update the knowledge base by fetching papers from arXiv or crawling web URLs.
    * Directly search arXiv for papers.
    * View retrieved context to understand the basis of the AI's answers.

    This tool is intended for researchers, students, and anyone looking to efficiently navigate and extract insights from academic documents.
    """)
    st.link_button("View Project on GitHub", "https://github.com/Fane-Nathan/Research-Assistant/tree/alpha-release", help="Opens the project's GitHub repository in a new tab.")


# --- Fetch Data Tab ---
with tab_fetch:
    st.header("⏬ Update Knowledge Base")
    st.markdown("Add new research papers from arXiv or crawl web URLs to expand the local knowledge base. This process involves fetching, chunking, embedding, and indexing the content.")

    # Initialize session state for fetch tab inputs if not present
    if 'fetch_arxiv_query' not in st.session_state: st.session_state.fetch_arxiv_query = "large language models"
    if 'fetch_num_arxiv' not in st.session_state: st.session_state.fetch_num_arxiv = 5 # Default to a smaller number for quicker tests
    if 'fetch_web_url' not in st.session_state: st.session_state.fetch_web_url = ""
    if 'fetch_suggest_sources' not in st.session_state: st.session_state.fetch_suggest_sources = False
    if 'fetch_topic_for_suggestion' not in st.session_state: st.session_state.fetch_topic_for_suggestion = "AI in healthcare"


    # UI for source selection
    st.subheader("Select Data Sources")
    
    # Section for arXiv
    with st.expander("📚 Fetch from arXiv", expanded=True):
        arxiv_query_fetch = st.text_input(
            "arXiv Query:", 
            value=st.session_state.fetch_arxiv_query, 
            key="fetch_arxiv_query_input",
            help="Enter keywords or topics to search on arXiv (e.g., 'transformer models', 'climate change AND policy')."
        )
        st.session_state.fetch_arxiv_query = arxiv_query_fetch
        
        num_arxiv_fetch = st.number_input(
            "Max arXiv Results to Fetch:", 
            min_value=0, 
            max_value=50, # Set a reasonable max to prevent very long fetches
            value=st.session_state.fetch_num_arxiv, 
            step=1, 
            key="fetch_num_arxiv_input",
            help="Number of papers to retrieve from arXiv for this query. Set to 0 to skip arXiv."
        )
        st.session_state.fetch_num_arxiv = num_arxiv_fetch

    # Section for Web URL
    with st.expander("🌐 Crawl a Web URL", expanded=True):
        web_url_fetch_val = st.text_input(
            "Single Web URL to Crawl (Optional):",
            value=st.session_state.fetch_web_url,
            key="fetch_web_url_input",
            placeholder="https://example.com/research-article.html",
            help="Provide a direct URL to a research paper or relevant webpage. The system will attempt to extract its content."
        )
        st.session_state.fetch_web_url = web_url_fetch_val
    
    # Suggest sources (kept as an option, but might be less used if manual inputs are primary)
    # use_suggestions = st.checkbox(
    #     "Suggest sources using LLM (experimental, may override above if successful)",
    #     value=st.session_state.get("fetch_suggest_sources", False),
    #     key="fetch_suggest_checkbox"
    # )
    # st.session_state.fetch_suggest_sources = use_suggestions
    # if use_suggestions:
    #     st.info("LLM-based source suggestion is not fully implemented in this UI version yet.", icon="🚧")
        # topic_for_suggestion = st.text_input(
        #     "Topic for LLM Source Suggestion:",
        #     value=st.session_state.fetch_topic_for_suggestion,
        #     key="fetch_topic_input",
        #     help="If using suggestions, provide a broad topic."
        # )
        # st.session_state.fetch_topic_for_suggestion = topic_for_suggestion


    st.markdown("---")
    fetch_data_button_disabled = not project_modules_loaded or (not st.session_state.fetch_arxiv_query and st.session_state.fetch_num_arxiv == 0 and not st.session_state.fetch_web_url)
    fetch_data_button_tooltip = ""
    if not project_modules_loaded:
        fetch_data_button_tooltip = "System modules not loaded. Refresh page."
    elif fetch_data_button_disabled: # Implies no sources selected
        fetch_data_button_tooltip = "Please provide an arXiv query, a web URL, or set number of arXiv results > 0."


    fetch_data_button = st.button(
        "🚀 Fetch and Process Data", 
        type="primary", 
        disabled=fetch_data_button_disabled, 
        help=fetch_data_button_tooltip or "Starts the data fetching, processing, and indexing pipeline.",
        use_container_width=True
        )
    
    if not project_modules_loaded:
        st.caption("🔧 **System initialization required** - Please refresh the page to enable data fetching.")

    if fetch_data_button and project_modules_loaded and setup_data_and_fetch_cli and config:
        with st.spinner("⏳ Fetching and processing data... This can take some time depending on the number of sources. Please wait."):
            try:
                # Prepare arguments for setup_data_and_fetch_cli
                # This dictionary approach is cleaner for constructing the Namespace
                fetch_cli_args_dict = {
                    "suggest_sources": st.session_state.get("fetch_suggest_sources", False), # Retained if CLI supports
                    "topic": st.session_state.get("fetch_topic_for_suggestion", "") if st.session_state.get("fetch_suggest_sources") else "",
                    "arxiv_query": st.session_state.get("fetch_arxiv_query", ""),
                    "num_arxiv": st.session_state.get("fetch_num_arxiv", 0),
                    "web_urls_to_process": [], # For CLI that accepts URLs directly
                    "num_web_pages": config.MAX_PAGES_TO_CRAWL if hasattr(config, 'MAX_PAGES_TO_CRAWL') else 1 # Default if not in config
                }

                # PYLANCE FIX for line 567: Ensure strip is called on a string.
                single_web_url_from_state = st.session_state.get("fetch_web_url", "")
                single_web_url = str(single_web_url_from_state or "").strip()
                
                if single_web_url:
                    fetch_cli_args_dict["web_urls_to_process"] = [single_web_url]
                
                # If setup_data_and_fetch_cli expects TARGET_WEB_URLS in config to be set:
                # This is less ideal than passing directly.
                # original_target_web_urls = list(config.TARGET_WEB_URLS) if hasattr(config, 'TARGET_WEB_URLS') else []
                # if single_web_url:
                #     config.TARGET_WEB_URLS = [single_web_url] # Or append/prepend as needed
                # else:
                #     config.TARGET_WEB_URLS = [] # Or keep original if no single URL provided and CLI uses it

                # Ensure arxiv_query is not empty if num_arxiv > 0 and not suggesting (if parser needs it)
                if not fetch_cli_args_dict["suggest_sources"] and \
                   not fetch_cli_args_dict["arxiv_query"] and \
                   fetch_cli_args_dict["num_arxiv"] > 0:
                    fetch_cli_args_dict["arxiv_query"] = "placeholder_for_cli_parser" 
                    logger.warning("Used placeholder arXiv query as num_arxiv > 0 but query was empty.")


                fetch_cli_args = argparse.Namespace(**fetch_cli_args_dict)
                logger.info(f"Calling setup_data_and_fetch_cli with args: {fetch_cli_args}")
                
                status_message = asyncio.run(setup_data_and_fetch_cli(fetch_cli_args))
                
                # Restore config if it was modified (if applicable)
                # if single_web_url or not single_web_url: # If config.TARGET_WEB_URLS was changed
                #    config.TARGET_WEB_URLS = original_target_web_urls

                st.success(f"✅ Data processing complete! Status: {status_message}")
                logger.info(f"Data fetch status from CLI: {status_message}")

                st.session_state.force_component_reload = True
                initialize_and_load_components() # Reload components to reflect new data
                
                if st.session_state.get('components_loaded_successfully', False):
                    st.toast("RAG Components reloaded with new data.", icon="🔄")
                    if st.session_state.get('knowledge_base_is_empty', True):
                        st.warning("Data fetched, but the knowledge base still seems empty or components didn't fully utilize it. Please check logs and data sources.", icon="⚠️")
                    else:
                        st.success("Knowledge base updated and RAG components are ready!", icon="👍")
                else:
                    st.warning("🔄 **Components couldn't reload automatically after data fetch.** Your data might have been fetched, but you may need to refresh the page to use it in the Recommend tab.", icon="⚠️")
                
                st.rerun() # Rerun to update UI elements based on new state

            except ImportError as e_imp: # Should be caught earlier, but good to have
                st.error("🔧 **Technical issue.** Data fetching system couldn't load. Refresh or check setup.", icon="❌")
                logger.error(f"ImportError during data fetch call: {e_imp}", exc_info=True)
            except Exception as e_fetch:
                st.error(f"📡 **Data fetch encountered an issue:** {str(e_fetch)[:300]}... This might be temporary. Please try again. If it persists, check logs or your inputs.", icon="🌐")
                logger.error(f"Data fetch error in Streamlit UI: {e_fetch}", exc_info=True)
    elif fetch_data_button and (not project_modules_loaded or not setup_data_and_fetch_cli or not config) :
         st.error("🚨 **Critical Error:** Data fetching system is not available (missing core modules/config). Check application setup.", icon="❌")


# --- Search arXiv Tab ---
with tab_arxiv:
    st.header("🔍 Search arXiv Directly")
    st.markdown("Use this tab to perform a quick search on arXiv.org. Results are shown directly and are not automatically added to the local knowledge base.")

    if 'arxiv_query_direct' not in st.session_state: st.session_state.arxiv_query_direct = "quantum machine learning"
    if 'arxiv_num_results_direct' not in st.session_state: st.session_state.arxiv_num_results_direct = 5

    arxiv_query_direct_val = st.text_input(
        "arXiv Search Query:", 
        value=st.session_state.arxiv_query_direct, 
        key="arxiv_direct_query_input",
        help="Enter your search terms for arXiv."
        )
    st.session_state.arxiv_query_direct = arxiv_query_direct_val

    arxiv_num_direct_val = st.number_input(
        "Max Results to Display:", 
        min_value=1, max_value=25, 
        value=st.session_state.arxiv_num_results_direct, 
        step=1, 
        key="arxiv_direct_num_input",
        help="How many search results to fetch and display from arXiv."
        )
    st.session_state.arxiv_num_results_direct = arxiv_num_direct_val
    
    submit_arxiv_search_disabled = not project_modules_loaded or not run_arxiv_search_cli or not st.session_state.arxiv_query_direct
    submit_arxiv_search = st.button(
        "Search arXiv", 
        type="primary", 
        disabled=submit_arxiv_search_disabled,
        help="Search arXiv with the provided query." if not submit_arxiv_search_disabled else "arXiv search function not ready or query is empty."
        )

    if submit_arxiv_search and project_modules_loaded and run_arxiv_search_cli:
        logger.info(f"Performing direct arXiv search for: '{st.session_state.arxiv_query_direct}', num_results={st.session_state.arxiv_num_results_direct}")
        with st.spinner("Searching arXiv..."):
            try:
                # run_arxiv_search_cli should return a list of dictionaries or similar structured data
                results = run_arxiv_search_cli(st.session_state.arxiv_query_direct, st.session_state.arxiv_num_results_direct)
                
                if results and isinstance(results, list):
                    st.subheader(f"Found {len(results)} results on arXiv for '{st.session_state.arxiv_query_direct}':")
                    for i, paper in enumerate(results):
                        # Assuming paper is a dict with keys like 'title', 'summary', 'authors', 'pdf_url', 'published'
                        with st.container():
                            st.markdown(f"##### {i+1}. {paper.get('title', 'N/A')}")
                            if paper.get('authors'):
                                authors_list = paper.get('authors', [])
                                st.markdown(f"**Authors:** {', '.join(authors_list) if isinstance(authors_list, list) else authors_list}")
                            if paper.get('published'):
                                st.markdown(f"**Published:** {paper.get('published', 'N/A')}")
                            if paper.get('summary'):
                                with st.expander("Abstract/Summary", expanded= (i < 2) ): # Expand first few
                                    st.markdown(paper.get('summary'))
                            
                            # PYLANCE FIX for line 927: Ensure pdf_url is a string.
                            pdf_url_val = paper.get('pdf_url')
                            if pdf_url_val and isinstance(pdf_url_val, str):
                                st.link_button("View PDF on arXiv", pdf_url_val, help=f"Open {paper.get('title', 'paper')} on arXiv")
                            elif pdf_url_val: # It exists but not a string
                                st.markdown(f"**PDF Link (raw):** {pdf_url_val} (Note: Expected a string URL)")
                            # else: No PDF URL, so no button is shown.
                            st.markdown("---")
                elif results: # Some result but not a list
                     st.warning(f"Received unexpected result format from arXiv search: {type(results)}. Cannot display.", icon="⚠️")
                     logger.warning(f"Unexpected arXiv search result type: {type(results)}, data: {results}")
                else:
                    st.info(f"No results found on arXiv for your query: '{st.session_state.arxiv_query_direct}'. Try different keywords.", icon="ℹ️")
            except Exception as e_arxiv_search:
                st.error(f"🔍 **arXiv search encountered an issue:** {str(e_arxiv_search)[:200]}...", icon="🌐")
                logger.error(f"Direct arXiv search error in UI: {e_arxiv_search}", exc_info=True)
    elif submit_arxiv_search and (not project_modules_loaded or not run_arxiv_search_cli):
        st.error("🚨 **Critical Error:** arXiv search functionality is not available (missing core modules).", icon="❌")


# --- Feedback Tab ---
with tab_feedback:
    st.header("📝 Submit Feedback or Report Issues")
    st.markdown("""
    Your feedback is valuable in improving this Research Assistant! 
    If you encounter any bugs, have suggestions for new features, or want to share your experience, please use the form below.
    """)
    feedback_form_url = "https://tally.so/embed/n0kkp6?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1"
    try:
        components.iframe(feedback_form_url, height=600, scrolling=True)
    except Exception as e_iframe:
        logger.error(f"Error embedding feedback form: {e_iframe}")
        st.markdown(f"""
        Oops! The embedded feedback form could not be loaded. 
        You can still submit feedback directly by visiting this link:
        [Feedback Form]({feedback_form_url.split('?')[0]})
        """)
        st.link_button("Open Feedback Form in New Tab", feedback_form_url.split('?')[0])

# --- Footer ---
st.divider()
st.markdown("<div style='text-align: center; font-size: 0.9em; color: #777;'>Research Paper Assistant | Powered by Streamlit & RAG</div>", unsafe_allow_html=True)

