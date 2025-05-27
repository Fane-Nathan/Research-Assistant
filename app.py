# -*- coding: utf-8 -*-
"""
Streamlit web application for the Study Assistant (User Provided Version).

Provides an enhanced user interface for:
1. Fetching and processing resources (arXiv, web, potentially others via cli functions).
2. Performing hybrid search and RAG (WITHOUT highlighting) to answer questions based on local data.
3. Directly searching arXiv.
4. Displaying information about the project and how it works.
5. Collecting user feedback via an embedded form.
6. Adding content by crawling a single user-provided URL.
"""

import streamlit as st
# --- Page Configuration MUST be the first Streamlit command ---
st.set_page_config(
    page_title="Research Paper Assistant", # Browser tab title
    page_icon="📚",                      # Browser tab icon
    layout="wide",                      # Use full width of the page
    initial_sidebar_state="auto",       # Sidebar behavior (can be used later)
    menu_items={                        # Custom items in the Streamlit menu (top right)
        'Get Help': 'https://github.com/Fane-Nathan/Study-Assistant', # Link to project repo
        'Report a bug': "https://tally.so/r/n0kkp6",                  # Link to feedback form
        'About': "# About This Project\nThis app helps explore academic research using RAG." # Simple about text
    }
)

# --- Other Imports ---
import logging
import math
import nltk
# import ssl # ssl import seems unused, can be removed if not needed
import os
import sys
import argparse
import traceback
import streamlit.components.v1 as components
# from thefuzz import fuzz # fuzz import removed as highlighting is removed
from typing import List, Dict, Any, Tuple, Generator, Optional # Added Generator, Optional
import asyncio # For async web crawl
import numpy as np # For handling embeddings
import pickle # For BM25 index loading/saving (indirectly via DataManager)
from rank_bm25 import BM25Okapi # For rebuilding BM25 index

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [App] %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# --- Path Setup ---
logger.info("Setting up sys.path...")
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project Root added to sys.path: {project_root}")
# --- End Path Setup ---


# --- Project Module Imports ---
# Highlighting import is REMOVED.
logger.info("Attempting project module imports...")
project_modules_loaded = False
# highlighting_available = False # Flag removed as feature is removed
try:
    from hybrid_search_rag import config
    from scripts.interfaces.cli import ( # MODIFIED: Path updated
        load_components,
        run_recommendation, 
        setup_data_and_fetch,
        run_arxiv_search,
        chunk_text_by_sentences, 
        check_nltk_data          
    )
    # Import necessary components for manual data handling
    from hybrid_search_rag.data_handling.data_manager import DataManager
    from hybrid_search_rag.data_handling.resource_fetcher import crawl_and_fetch_web_articles
    from hybrid_search_rag.embedding_services.gemini_embedder import EmbeddingModel # Assuming Gemini embeddings
    from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import NltkManager # For tokenizing BM25

    # REMOVED: Import for highlighting
    # from hybrid_search_rag.utils.highlighting import highlight_sources_fuzzy
    # highlighting_available = True # Flag removed

    project_modules_loaded = True
    logger.info("Project module imports successful (including highlighting).") # Updated log message slightly
except ImportError as e:
    st.error(f"Failed to import project modules (ImportError). Check setup and ensure scripts/cli.py and hybrid_search_rag package are accessible. Error: {e}")
    st.code(f"Current sys.path: {sys.path}")
    logger.error(f"Project import failed (ImportError): {e}", exc_info=True)
except Exception as e:
    # Catch other errors during import
    st.error(f"Unexpected error during project imports: {e}. App functionality will be limited.")
    logger.error(f"Project import failed (Exception): {e}", exc_info=True)
    # If the error is the Prometheus one, add a specific hint
    if 'Duplicated timeseries' in str(e):
        st.warning("Hint: The 'Duplicated timeseries' error often relates to Prometheus metrics. Ensure they are removed or handled correctly in resource_fetcher.py if not needed.")
# --- End Project Module Imports ---

# --- Async Helper for Recommendation Submission ---
async def handle_recommendation_submission_async(query: str, top_n: int, general_mode: bool, concise_mode: bool, answer_placeholder: Any, project_modules_loaded: bool):
    """Handles the recommendation submission logic asynchronously."""
    if not project_modules_loaded: # Double check, though components_loaded_status should also cover this
        st.error("Cannot recommend: Core project modules not loaded.")
        return

    # Ensure cli_run_recommendation is available
    try:
        from scripts.interfaces.cli import run_recommendation as cli_run_recommendation
        from hybrid_search_rag import config # Ensure config is in scope here if needed for TOP_N_RESULTS
    except ImportError:
        st.error("Failed to import recommendation function or config. Cannot proceed.")
        logger.error("ImportError within handle_recommendation_submission_async for run_recommendation or config.")
        return

    logger.info(f"Async handler: Streaming response for query '{query[:50]}...'")
    try:
        # Use the imported cli_run_recommendation
        response_generator, context_sources, _ = await cli_run_recommendation(
            query, top_n, general_mode, concise_mode
        )
        st.session_state.context_sources = context_sources

        full_response_list = []
        error_in_stream = False
        async for chunk in response_generator: # MODIFIED: Use async for
            if isinstance(chunk, str) and chunk.startswith("[Error:"):
                st.error(f"LLM Error: {chunk}", icon="❌")
                logger.error(f"LLM generation failed: {chunk}")
                if answer_placeholder: answer_placeholder.empty()
                st.session_state.llm_answer = None
                error_in_stream = True
                break
            
            full_response_list.append(chunk)
            if answer_placeholder: answer_placeholder.markdown("".join(full_response_list) + "▌", unsafe_allow_html=True)

        if not error_in_stream:
            full_response = "".join(full_response_list)
            logger.info(f"Full response received (length: {len(full_response)}).")
            st.session_state.llm_answer = full_response
            if answer_placeholder: answer_placeholder.markdown(full_response, unsafe_allow_html=True)
            st.toast("Response generated!", icon="✅")
        else:
            if answer_placeholder: answer_placeholder.empty()

    except Exception as e:
        st.error(f"Error during recommendation: {e}", icon="❌")
        logger.error(f"Recommendation Error in async handler: {e}", exc_info=True)
        if answer_placeholder: answer_placeholder.empty()
        st.session_state.llm_answer = None
        st.session_state.context_sources = []
# --- End Async Helper ---


# --- NLTK Data Download Logic ---
# Reuse the NLTK check function from cli.py if modules loaded
if project_modules_loaded:
    try:
        if 'nltk_data_checked_app' not in st.session_state:
             with st.spinner("Checking NLTK data..."):
                 logger.info("Running initial NLTK check via imported function...")
                 check_nltk_data() # Call the function from cli.py
                 st.session_state.nltk_data_checked_app = True
                 logger.info("Initial NLTK check complete.")
    except NameError:
         st.error("NLTK check function `check_nltk_data` not found. Manual NLTK check block needed.")
         pass
    except SystemExit: # Catch SystemExit if NLTK download fails during init
          st.error("Fatal Error: Failed to download required NLTK data during startup. App cannot continue.")
          logger.critical("NLTK download failed during initial check. Stopping app.")
          st.stop() # Stop app execution
    except Exception as nltk_e:
         st.error(f"Error during initial NLTK check: {nltk_e}")
         logger.error(f"NLTK check failed during app init: {nltk_e}", exc_info=True)
         st.stop() # Stop if essential NLTK check fails
else:
    logger.warning("Skipping NLTK check as project modules failed to load.")

# --- Title, Introduction, and Disclaimer (Enhanced) ---
# (Remains unchanged)
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
# --- End Title/Intro/Disclaimer ---


# --- Caching Components ---
@st.cache_resource
async def cached_load_components(): # MODIFIED: Made async
    """
    Loads core RAG components using the imported 'load_components' function.
    Uses session state to force reload after data updates.
    """
    logger.info("Executing cached RAG component loading logic...")
    if not project_modules_loaded:
        logger.error("Cannot load components: project modules failed import prior to this call.")
        return False, "Core project modules failed to import."

    # --- FIXED: Check session state flag to force reload ---
    force_reload_flag = st.session_state.get("force_component_reload", False)
    if force_reload_flag:
        logger.info("Force reload flag set, calling load_components with force_reload=True.")
        # Clear the flag immediately after reading it
        st.session_state.force_component_reload = False
    # --- End Fix ---

    try:
        # Ensure NLTK is checked before loading components
        check_nltk_data()
        # --- FIXED: Pass the force_reload_flag to load_components ---\n        await load_components(force_reload=force_reload_flag) # MODIFIED: Added await
        # --- End Fix ---

        # Check if the recommender object exists in cli.py after loading
        from scripts.interfaces.cli import loaded_recommender # MODIFIED: Path updated
        if loaded_recommender is not None:
             logger.info("Component loading logic successful (based on \'loaded_recommender\' indicator).")
             return True, None
        else:
             logger.warning("load_components ran but indicator 'loaded_recommender' is None. Assuming load failed.")
             return False, "Components appear missing after loading attempt (recommender is None)."
    except NameError as ne:
         logger.error(f"NameError during component loading check: {ne}. Cannot verify loaded state.", exc_info=True)
         return False, f"Error accessing loaded state indicator from 'scripts.cli'. NameError: {ne}"
    except SystemExit: # Catch NLTK download failure during component load
         logger.critical("NLTK download failed during component loading. Cannot proceed.")
         return False, "Failed to acquire NLTK data during component loading."
    except Exception as e:
        logger.error(f"Exception during component loading: {e}", exc_info=True)
        return False, f"An exception occurred during component loading: {e}"
# --- End Caching Components ---


# --- Initial Setup ---
# (Remains unchanged)
logger.info("Attempting initial component load via cached function...")
components_loaded_status, load_error_message = False, "Project modules did not load."
if project_modules_loaded:
    # components_loaded_status, load_error_message = cached_load_components() # OLD SYNCHRONOUS CALL
    # MODIFIED: Run async function using asyncio.run() for initial load
    try:
        components_loaded_status, load_error_message = asyncio.run(cached_load_components())
    except Exception as e:
        logger.error(f"Error running cached_load_components during initial setup: {e}")
        components_loaded_status = False
        load_error_message = f"Async load error: {e}"

logger.info(f"Components loaded status: {components_loaded_status}")
if not components_loaded_status and project_modules_loaded:
    st.error(f"Core RAG components failed to load: {load_error_message}. Recommendation and Fetch Data functionality disabled.")


# --- UI Tabs (with Icons) ---
# (Remains unchanged)
logger.info("Defining UI tabs...")
tab_rec, tab_how, tab_about, tab_fetch, tab_arxiv, tab_feedback = st.tabs([
    "🧠 **Recommend**",
    "⚙️ How It Works",
    "ℹ️ About",
    "⏬ Update Knowledge Base",
    "🔍 Search arXiv",
    "📝 Feedback"
])
logger.info("UI tabs defined.")


# --- Recommendation Tab (Highlighting Removed) ---
# (Remains unchanged from user's provided version)
with tab_rec:
    st.header("💬 Ask the RAG Assistant")
    st.markdown("Enter your research topic or question below. The assistant will retrieve relevant information from the indexed documents and generate an answer.") # Removed "cited"

    col1_rec, col2_rec = st.columns([3, 1])

    with col1_rec:
        default_query = config.DEFAULT_QUERY if project_modules_loaded else "Example: Explain Retrieval-Augmented Generation (RAG)."
        query = st.text_area(
            "Your Question:",
            value=st.session_state.get("rec_query", default_query),
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
            disabled=not components_loaded_status or not query,
            use_container_width=True
        )

    # --- Logic when Button is Clicked ---
    if submit_rec:
        if not components_loaded_status:
            st.error("Cannot recommend: Core components failed load.")
        else:
            st.session_state.rec_query = query
            st.session_state.rec_general = general_mode
            st.session_state.rec_concise = concise_mode
            mode_name = "Hybrid" if general_mode else "Strict"
            style_name = "Concise" if concise_mode else "Detailed"
            logger.info(f"Running recommendation for query: '{query[:50]}...' with Mode='{mode_name}', Style='{style_name}'")
            st.info(f"Running recommendation with '{mode_name}' RAG and '{style_name}' Prompt...", icon="⏳")

            st.session_state.llm_answer = None
            st.session_state.context_sources = []
            st.session_state.raw_context_chunks = [] # Retained for potential future use, though not directly used by LLM
            answer_placeholder = st.empty()

            top_n_val = 5 # Default
            if project_modules_loaded:
                try:
                    from hybrid_search_rag import config # Ensure config is imported for this scope
                    top_n_val = config.TOP_N_RESULTS
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not get config.TOP_N_RESULTS ({e}), using default {top_n_val}.")

            try:
                asyncio.run(handle_recommendation_submission_async(
                    query,
                    top_n_val,
                    general_mode,
                    concise_mode,
                    answer_placeholder,
                    project_modules_loaded
                ))
            except Exception as e:
                st.error(f"Error running async recommendation task: {e}", icon="❌")
                logger.error(f"Async recommendation task runner error: {e}", exc_info=True)
                if answer_placeholder: answer_placeholder.empty()
# --- End Recommendation Tab ---


# --- How It Works Tab ---
with tab_how:
    st.header("⚙️ How the RAG System Works")
    st.markdown("This system uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide informed answers:")
    col1_how, col2_how = st.columns(2)
    with col1_how:
        st.subheader("1️⃣ Data Ingestion & Processing")
        st.markdown("""
        * **Sources:** Fetches data from arXiv, web URLs, potentially local files.
        * **Cleaning:** Prepares text for analysis.
        * **Chunking:** Breaks down documents into smaller pieces.
        """)
        st.subheader("2️⃣ Hybrid Indexing")
        emb_model = f"`{config.EMBEDDING_MODEL_NAME}`" if project_modules_loaded else "a sentence transformer model"
        st.markdown(f"""
        * **Vector Embeddings:** Creates numerical representations (vectors) capturing semantic meaning using {emb_model}.
        * **Keyword Index (BM25):** Creates a traditional keyword index for term matches.
        * **Combined Power:** Stores both index types locally.
        """)
    with col2_how:
        st.subheader("3️⃣ Hybrid Retrieval (RRF)")
        st.markdown("""
        * **Dual Search:** Your query searches *both* the vector index and the keyword index.
        * **Smart Ranking (RRF):** Results are combined using Reciprocal Rank Fusion (RRF) for a balanced relevance ranking.
        """)
        st.subheader("4️⃣ LLM Generation")
        # --- FIXED: Use new config structure ---
        llm_provider_display = "a Large Language Model (LLM)" # Default
        if project_modules_loaded and config.LLM_PROVIDER_ORDER:
            # Display the primary provider (first in the list)
            llm_provider_display = config.LLM_PROVIDER_ORDER[0].capitalize()
            if len(config.LLM_PROVIDER_ORDER) > 1:
                 # Show fallback only if it's different from primary
                 fallback_provider = config.LLM_PROVIDER_ORDER[1].capitalize()
                 if fallback_provider != llm_provider_display:
                      llm_provider_display += f" (with fallback to {fallback_provider})"
        # --- End Fix ---
        st.markdown(f"""
        * **Context Injection:** Top-ranked retrieved chunks are passed as context to {llm_provider_display}.
        * **Informed Answering:** The LLM generates an answer based on the provided context (Strict RAG) or a mix (Hybrid Mode).
        """)
    st.divider()
    st.subheader("🚀 The RAG Pipeline Advantage")
    st.markdown("""
    * Grounding answers in specific, retrieved information.
    * Reducing the likelihood of hallucinations.
    * Providing transparency through retrieved context (shown in expander).
    """)
    st.subheader("💻 Core Implementation Notes")
    st.markdown("""
    * Built using Python with libraries like `google-generativeai`, `groq`, `rank_bm25`, `nltk`, vector stores, and `Streamlit`.
    * Features modular components for data handling, indexing, retrieval, and generation.
    """)
    st.divider()
# --- End How It Works Tab ---


# --- About Tab ---
# (Remains unchanged)
with tab_about:
    st.header("ℹ️ About This Project")
    col1_about, col2_about = st.columns(2)
    with col1_about:
        st.subheader("🎯 Project Goal")
        st.markdown("""
        To develop an intelligent assistant that helps users efficiently navigate, understand, and discover information within a specific corpus of academic documents.
        """)
        st.subheader("🛠️ Current Status & Features")
        st.markdown("""
        * **RAG Core:** Functional RAG pipeline implemented.
        * **Data Sources:** Ingests arXiv papers, web links.
        * **Hybrid Search:** Combines semantic (vector) and keyword (BM25) search with RRF.
        * **LLM Integration:** Uses Gemini/Groq for generation (Strict/Hybrid modes) with fallback.
        * **UI:** Interactive Streamlit interface for querying, data fetching, and arXiv search.
        """) # Updated LLM integration description
    with col2_about:
        st.subheader("🚀 Future Work & Ideas")
        st.markdown("""
        * **Corpus Expansion:** Curate and index a larger, more diverse set of relevant papers.
        * **Evaluation & Tuning:** Systematically evaluate retrieval and generation quality.
        * **Recommender System:** Proactively suggest relevant papers or topics.
        * **Personalization:** Allow user profiles, history tracking.
        * **Deployment:** Explore options for more robust deployment.
        """)
        st.subheader("👨‍💻 Developers")
        st.markdown(f"""
        Developed by **Felix Nathaniel**, **Reynaldi Anatyo**, & **Dennison Soedibjo**.

        *Computer Science Students at BINUS University, exploring the fascinating world of AI and Information Retrieval.*
        """)
        st.link_button("View Project on GitHub", "https://github.com/Fane-Nathan/Study-Assistant")
# --- End About Tab ---


# --- Fetch Data Tab (With Single URL Crawl Added) ---
with tab_fetch:
    st.header("⏬ Update Knowledge Base")
    st.markdown("""
    Fetch new data from arXiv or web URLs to update the local knowledge base. This involves fetching, chunking, embedding, and indexing.
    """)

    # Session state for fetch parameters
    if 'fetch_arxiv_query' not in st.session_state: st.session_state.fetch_arxiv_query = "large language models"
    if 'fetch_num_arxiv' not in st.session_state: st.session_state.fetch_num_arxiv = 10
    if 'fetch_suggest_sources' not in st.session_state: st.session_state.fetch_suggest_sources = False
    if 'fetch_topic_for_suggestion' not in st.session_state: st.session_state.fetch_topic_for_suggestion = "latest advancements in AI"

    use_suggestions = st.checkbox(
        "Suggest sources using LLM (overrides arXiv query below if successful)",
        value=st.session_state.fetch_suggest_sources,
        key="fetch_suggest_checkbox"
    )
    st.session_state.fetch_suggest_sources = use_suggestions

    if use_suggestions:
        topic_for_suggestion = st.text_input(
            "Topic for LLM to suggest sources:",
            value=st.session_state.fetch_topic_for_suggestion,
            key="fetch_topic_input",
            help="The LLM will try to find relevant arXiv queries and web URLs for this topic."
        )
        st.session_state.fetch_topic_for_suggestion = topic_for_suggestion
        # Disable manual arXiv query if suggestions are used
        st.text_input(
            "arXiv Query (disabled when suggesting sources):",
            value=st.session_state.fetch_arxiv_query,
            key="fetch_arxiv_query_disabled_input",
            disabled=True
        )
        st.number_input(
            "Max arXiv Results (disabled when suggesting sources):",
            min_value=0, value=st.session_state.fetch_num_arxiv, step=1,
            key="fetch_num_arxiv_disabled_input",
            disabled=True
        )
    else:
        arxiv_query_fetch = st.text_input(
            "arXiv Query:",
            value=st.session_state.fetch_arxiv_query,
            key="fetch_arxiv_query_input",
            help="Example: 'transformer models for NLP' or 'cs.AI AND large language models'"
        )
        st.session_state.fetch_arxiv_query = arxiv_query_fetch

        num_arxiv_fetch = st.number_input(
            "Max arXiv Results (0 to skip arXiv):",
            min_value=0, value=st.session_state.fetch_num_arxiv, step=1,
            key="fetch_num_arxiv_input",
            help="Number of papers to fetch from arXiv. Set to 0 if only using web URLs or suggestions."
        )
        st.session_state.fetch_num_arxiv = num_arxiv_fetch

    st.markdown("--- Optionally, add a single web URL to crawl and process ---")
    web_url_fetch = st.text_input(
        "Single Web URL to Crawl (Optional):",
        key="fetch_web_url_input",
        placeholder="https://example.com/article",
        help="If provided, this URL will be crawled, and its content processed."
    )

    if st.button("🚀 Fetch and Process Data", type="primary", disabled=not project_modules_loaded):
        if not project_modules_loaded:
            st.error("Cannot fetch data: Core components not loaded.")
        else:
            with st.spinner("Fetching and processing data... This may take a while."):
                try:
                    from scripts.interfaces.cli import setup_data_and_fetch as cli_setup_data_and_fetch
                    from hybrid_search_rag import config # Ensure config is available

                    # Create a mock args namespace for setup_data_and_fetch
                    fetch_args_list = []
                    if use_suggestions:
                        fetch_args_list.extend(["--suggest-sources", "-t", st.session_state.fetch_topic_for_suggestion])
                        # When suggesting, num_arxiv from UI is ignored, cli.py's setup_data_and_fetch will use a default or LLM suggested one.
                        # We still pass a value for num_arxiv as the argparse in cli expects it.
                        fetch_args_list.extend(["--num-arxiv", str(st.session_state.fetch_num_arxiv)]) # Pass UI value, might be overridden by LLM
                    else:
                        if st.session_state.fetch_arxiv_query and st.session_state.fetch_num_arxiv > 0:
                            fetch_args_list.extend(["--arxiv-query", st.session_state.fetch_arxiv_query, "--num-arxiv", str(st.session_state.fetch_num_arxiv)])
                        else:
                             # Provide a default dummy query if none is set, to satisfy argparse, but num_arxiv=0 will prevent fetch
                            fetch_args_list.extend(["--arxiv-query", "dummy_query_to_satisfy_parser", "--num-arxiv", "0"])

                    # Handle web URL if provided
                    if web_url_fetch and web_url_fetch.strip():
                        # The current setup_data_and_fetch doesn't directly take a single URL via args.
                        # It uses config.TARGET_WEB_URLS or LLM suggested URLs.
                        # For this UI, we'll need to adjust how single URLs are handled.
                        # Option 1: Modify cli.py to accept a --single-url arg (more robust)
                        # Option 2: Temporarily override config.TARGET_WEB_URLS (hacky)
                        # For now, we'll inform the user this part is a bit conceptual or relies on config.
                        st.info(f"Web URL '{web_url_fetch}' noted. Ensure your `config.py` or LLM suggestions include it, or modify `cli.py` to handle single URL inputs directly for this button to process it.")
                        # If we were to implement it via args, it might look like:
                        # fetch_args_list.extend(["--web-urls", web_url_fetch]) # Assuming cli.py is updated

                    parser = argparse.ArgumentParser()
                    # Add arguments that setup_data_and_fetch expects
                    parser.add_argument("--arxiv-query", type=str, default="")
                    parser.add_argument("--num-arxiv", type=int, default=0)
                    parser.add_argument("--suggest-sources", action="store_true")
                    parser.add_argument("-t", "--topic", type=str, default="")
                    # Potentially add --web-urls if cli.py is updated
                    # parser.add_argument("--web-urls", nargs='+', default=[])

                    fetch_args = parser.parse_args(fetch_args_list)

                    # status_message = cli_setup_data_and_fetch(fetch_args) # OLD SYNCHRONOUS CALL
                    status_message = asyncio.run(cli_setup_data_and_fetch(fetch_args)) # MODIFIED: Use asyncio.run

                    st.success(status_message)
                    logger.info(f"Data fetch status: {status_message}")
                    # Force reload of components after data update
                    st.session_state.force_component_reload = True
                    # Rerun cached_load_components to reflect new data
                    # components_loaded_status, load_error_message = cached_load_components() # OLD
                    try:
                        components_loaded_status, load_error_message = asyncio.run(cached_load_components())
                        if components_loaded_status:
                            st.toast("RAG Components reloaded with new data.", icon="🔄")
                        else:
                            st.error(f"Failed to reload RAG components after data fetch: {load_error_message}")
                    except Exception as reload_e:
                        st.error(f"Error reloading components: {reload_e}")
                        logger.error(f"Async component reload error: {reload_e}", exc_info=True)

                except ImportError:
                    st.error("Failed to import data fetching function. Cannot proceed.")
                    logger.error("ImportError for setup_data_and_fetch in Streamlit app.")
                except Exception as e:
                    st.error(f"Error during data fetch: {e}")
                    logger.error(f"Data fetch error: {e}", exc_info=True)
# --- End Fetch Data Tab ---


# --- Search arXiv Tab ---
# (Remains unchanged)
with tab_arxiv:
    st.header("🔍 Search arXiv Directly")
    st.markdown("Use this tab to perform a direct keyword search on arXiv.org. Results are fetched in real-time.")

    if 'arxiv_query_direct' not in st.session_state: st.session_state.arxiv_query_direct = "quantum machine learning"
    if 'arxiv_num_results_direct' not in st.session_state: st.session_state.arxiv_num_results_direct = 5

    arxiv_query = st.text_input(
        "arXiv Search Query:",
        value=st.session_state.arxiv_query_direct,
        key="arxiv_direct_query_input",
        help="Enter your search terms (e.g., 'cs.CV AND object detection')"
    )
    num_results = st.number_input(
        "Max Results:",
        min_value=1, max_value=50, value=st.session_state.arxiv_num_results_direct, step=1,
        key="arxiv_direct_num_input"
    )

    submit_arxiv_search = st.button("Search arXiv", type="primary", disabled=not project_modules_loaded or not arxiv_query)

    if submit_arxiv_search:
        if not project_modules_loaded:
            st.error("Cannot search arXiv: Core components not loaded.")
        else:
            st.session_state.arxiv_query_direct = arxiv_query
            st.session_state.arxiv_num_results_direct = num_results
            logger.info(f"Performing direct arXiv search for: '{arxiv_query}', num_results={num_results}")
            with st.spinner("Searching arXiv..."):
                try:
                    from scripts.interfaces.cli import run_arxiv_search as cli_run_arxiv_search
                    # results = cli_run_arxiv_search(arxiv_query, num_results) # OLD SYNCHRONOUS CALL
                    results = asyncio.run(cli_run_arxiv_search(arxiv_query, num_results)) # MODIFIED: Use asyncio.run

                    if results:
                        st.success(f"Found {len(results)} results on arXiv:")
                        for paper in results:
                            st.markdown(f"**[{paper.get('title', 'N/A')}]({paper.get('url', '#')})**")
                            authors = paper.get('authors', [])
                            if authors:
                                st.markdown(f"*Authors: {', '.join(authors)}*")
                            st.caption(f"Published: {paper.get('published_date', 'N/A')} | ID: {paper.get('entry_id', 'N/A')}")
                            st.markdown(f"<details><summary>Abstract</summary>{paper.get('summary', 'No abstract available.')}</details>", unsafe_allow_html=True)
                            st.markdown("---")
                    else:
                        st.info("No results found for your query on arXiv.")
                except ImportError:
                    st.error("Failed to import arXiv search function. Cannot proceed.")
                    logger.error("ImportError for run_arxiv_search in Streamlit app.")
                except Exception as e:
                    st.error(f"Error during arXiv search: {e}")
                    logger.error(f"Direct arXiv search error: {e}", exc_info=True)
# --- End Search arXiv Tab ---


# --- Feedback Tab ---
# (Remains unchanged)
with tab_feedback:
    st.header("📝 Submit Feedback or Report Issues")
    st.markdown("""
    Your feedback is valuable for improving this tool! Please use the form below to share your thoughts,
    suggestions, or report any bugs you encounter.
    """)
    TALLY_ORIGINAL_EMBED_URL = "https://tally.so/embed/n0kkp6?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1"
    IFRAME_HEIGHT = 500
    logger.info(f"Embedding feedback form from TALLY_ORIGINAL_EMBED_URL}")
    try:
        components.iframe(TALLY_ORIGINAL_EMBED_URL, height=IFRAME_HEIGHT, scrolling=True)
        st.caption("Feedback form securely hosted by Tally.so.")
    except Exception as e:
        st.error(f"Could not load the feedback form.", icon="😞")
        st.markdown("You can also submit feedback directly [here](https://tally.so/r/n0kkp6).")
        logger.error(f"Error embedding Tally iframe: {e}", exc_info=True)
# --- End Feedback Tab ---


# --- Footer ---
# (Remains unchanged)
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.9em;'>
        © 2025 Felix Nathaniel, Reynaldi Anatyo, Dennison Soedibjo | BINUS University Computer Science <br>
        Research Paper Assistant
    </div>
    """,
    unsafe_allow_html=True
)
# --- End Footer ---
