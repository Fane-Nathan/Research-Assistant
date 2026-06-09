# -*- coding: utf-8 -*-
"""
Streamlit Web Application: Research Paper Assistant (Full Code, Final Version)

This script contains the complete and final code for a redesigned, elegant UI with all features,
including the fix for the AttributeError on context display.
"""

# --- Core Imports ---
import json
import streamlit as st
import os
import sys
import asyncio
import time
import types
import argparse
import re
from typing import Dict, Any, Generator, AsyncGenerator

# --- Page Configuration: Must be the first Streamlit command ---
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/Fane-Nathan/Research-Assistant/tree/alpha-release',
        'Report a bug': "https://tally.so/r/n0kkp6",
        'About': "# Research Paper Assistant\nThis app uses RAG to explore academic literature."
    }
)

# --- Path and Module Setup ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from hybrid_search_rag import config
    from hybrid_search_rag.retrieval_algorithm.hybrid_recommender import RecommendationParams
    from hybrid_search_rag.text_processing.text_cleaner import clean_context_list
    from hybrid_search_rag.evaluation.retrieval_evaluator import RetrievalEvaluator, RetrievalEvaluationMetrics
    from hybrid_search_rag.retrieval_algorithm.reranker import ReRanker
    from scripts.cli import (
        load_components as load_components_cli,
        setup_data_and_fetch as setup_data_and_fetch_cli,
        run_arxiv_search as run_arxiv_search_cli,
        run_recommendation as cli_run_recommendation_async,
        check_nltk_data as check_nltk_data_cli,
    )
except Exception as e:
    st.error(f"A critical project module failed to import: `{e}`. The application cannot continue.")
    st.stop()


# --- Custom CSS for Elegant Styling ---
st.markdown("""
<style>
    /* ... (CSS from previous version remains unchanged) ... */
    .stApp { background-color: #020617; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF; }
    .st-emotion-cache-1s4o3wo p { color: #E2E8F0; }
    p, .st-emotion-cache-16txtl3.e1f1d6gn0 > p { color: #E2E8F0; }
    header[data-testid="stHeader"] { display: none !important; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
    .st-emotion-cache-p5msec { color: #FFFFFF; }
    .st-emotion-cache-1hver8f { background-color: #0F172A; }
    small { line-height: 1.5; color: #94A3B8; }
    .metadata-line { font-size: 0.85em; color: #94A3B8; margin-top: 12px; border-top: 1px solid #1E293B; padding-top: 8px; }
    .metadata-line a { color: #7DD3FC; text-decoration: none; font-weight: 500; }
    .metadata-line a:hover { text-decoration: underline; }
    .stButton>button { border-radius: 8px; border: 1px solid #38BDF8; background-color: #38BDF8; color: #020617; transition: all 0.2s ease-in-out; font-weight: 600; }
    .stButton>button:hover { background-color: #0F172A; color: #38BDF8; border-color: #38BDF8; }
    .stButton>button:disabled { background-color: #1E293B; color: #475569; border-color: #1E293B; cursor: not-allowed; }
    .st-emotion-cache-163ttbj { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
    [data-testid="stMetric"] { background-color: transparent !important; border: 0px !important; }
    [data-testid="stMetric"] label, [data-testid="stMetric"] div { color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

# --- Core Logic & Helper Functions ---
def format_expander_title(source: dict) -> str:
    title = source.get('title', 'Untitled Source')
    match = re.search(r'\[No Title - ID: http://arxiv.org/abs/([^v]+)v\d+_chunk_(\d+)\]', title)
    if match:
        arxiv_id, chunk_num = match.group(1), match.group(2)
        return f"📄 Source: arXiv:{arxiv_id} (Chunk {chunk_num})"
    return f"📄 {title[:75]}..." if len(title) > 75 else f"📄 {title}"

def initialize_and_load_components():
    force_reload = st.session_state.get("force_component_reload", False)
    st.session_state.component_load_error = None
    try:
        with st.spinner("Initializing RAG system components..."):
            loaded_data_dict = load_components_cli(force_reload=force_reload)
        st.session_state.force_component_reload = False
        if loaded_data_dict:
            st.session_state.rag_components = loaded_data_dict
            st.session_state.components_loaded_successfully = bool(loaded_data_dict.get("recommender"))
            st.session_state.knowledge_base_is_empty = loaded_data_dict.get("knowledge_base_is_empty", True)
        else:
            st.session_state.components_loaded_successfully = False
            st.session_state.knowledge_base_is_empty = True
    except Exception as e:
        st.session_state.force_component_reload = False
        st.session_state.components_loaded_successfully = False
        st.session_state.knowledge_base_is_empty = True
        st.session_state.component_load_error = str(e)

def run_evaluation_in_app():
    st.session_state.evaluation_running = True
    st.session_state.evaluation_results = None
    try:
        with st.status("Executing RAG evaluation...", expanded=True) as status:
            eval_dataset_path = "data_store/evaluation_sets/evaluation_dataset_llm_labeled.json"
            status.update(label=f"Loading dataset: `{eval_dataset_path}`")
            with open(eval_dataset_path, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            evaluation_dataset = [{"query": item["query_text"], "relevant_docs": set(item["relevant_doc_ids"])} for item in eval_data]
            rag_components, recommender, data_manager = st.session_state.rag_components, st.session_state.rag_components['recommender'], st.session_state.rag_components['data_manager']
            status.update(label="Initializing evaluation modules...")
            reranker, evaluator = ReRanker(), RetrievalEvaluator(recommender)
            resource_metadata, resource_embeddings, bm25_index = rag_components.get('metadata'), rag_components.get('embeddings'), rag_components.get('bm25_index')
            params, final_metrics = RecommendationParams(top_n_final=20, top_n_rerank=10, semantic_candidates=50, keyword_candidates=50, fusion_k=20), RetrievalEvaluationMetrics()
            total_queries, start_time = len(evaluation_dataset), time.time()
            progress_bar = st.progress(0, text=f"Starting evaluation of {total_queries} queries...")
            for i, item in enumerate(evaluation_dataset):
                query, relevant_docs_set = item["query"], item["relevant_docs"]
                candidate_results = recommender.recommend(query=query, params=params, resource_metadata=resource_metadata, resource_embeddings=resource_embeddings, bm25_index=bm25_index)
                candidate_docs = [doc for doc, _ in candidate_results]
                reranked_results = reranker.rerank(query, candidate_docs)
                retrieved_ids = [doc.get('entry_id') for doc, _ in reranked_results[:params.top_n_rerank]]
                hit_rate, mrr, ndcg = evaluator._calculate_ranking_metrics(retrieved_ids, relevant_docs_set)
                final_metrics.update_ranking_metrics(hit_rate, mrr, ndcg)
                precision, recall, f1 = evaluator._calculate_classification_metrics(retrieved_ids, relevant_docs_set)
                final_metrics.update_classification_metrics(precision, recall, f1)
                progress_bar.progress((i + 1) / total_queries, text=f"Processing query {i+1}/{total_queries}...")
            avg_metrics = final_metrics.get_average_metrics()
            avg_metrics['total_time_seconds'] = time.time() - start_time
            st.session_state.evaluation_results = avg_metrics
            status.update(label="Evaluation complete!", state="complete")
    except Exception as e:
        st.error(f"An error occurred during evaluation: {e}")
        st.session_state.evaluation_results = {"error": str(e)}
    finally:
        st.session_state.evaluation_running = False

async def handle_recommendation_submission_async(query_text: str, **kwargs):
    st.session_state.recommendation_processing = True
    st.session_state.llm_answer, st.session_state.context_sources = "", []
    answer_placeholder = kwargs.pop("answer_placeholder")
    current_answer = ""
    try:
        result = await cli_run_recommendation_async(query=query_text, **kwargs)
        if result:
            answer_stream, source_ids, source_documents = result
            
            if source_documents and isinstance(source_documents, list) and len(source_documents) > 0 and isinstance(source_documents[0], list):
                st.session_state.context_sources = source_documents[0]
            else:
                st.session_state.context_sources = source_documents if source_documents else []

            if isinstance(answer_stream, (types.GeneratorType, AsyncGenerator)):
                for token in answer_stream:
                    current_answer += token
                    answer_placeholder.markdown(current_answer + "▌")
                st.session_state.llm_answer = current_answer
            else:
                st.session_state.llm_answer = str(answer_stream)
            answer_placeholder.markdown(st.session_state.llm_answer)
        else:
            st.session_state.llm_answer = "No recommendation could be generated."
            answer_placeholder.info(st.session_state.llm_answer)
    except Exception as e:
        st.session_state.llm_answer = f"🤖 An error occurred: {e}"
        answer_placeholder.error(st.session_state.llm_answer)
    finally:
        st.session_state.recommendation_processing = False
        st.rerun()

# --- UI Rendering Functions ---

def render_sidebar():
    with st.sidebar:
        st.title("🔬 System Control")
        st.markdown("Configure the assistant's behavior and monitor system status.")
        st.markdown("---")
        with st.container(border=True):
            st.subheader("System Status")
            is_ready = st.session_state.get('components_loaded_successfully', False)
            is_empty = st.session_state.get('knowledge_base_is_empty', True)
            status_icon = "✅" if is_ready else "⚠️"
            kb_icon = "📚" if not is_empty else "📭"
            st.markdown(f"**RAG Status:** {status_icon} {'Online' if is_ready else 'Offline'}")
            st.markdown(f"**Knowledge Base:** {kb_icon} {'Populated' if not is_empty else 'Empty'}")
            if st.button("🔄 Reload RAG Components", use_container_width=True):
                st.session_state.force_component_reload = True
                st.rerun()
            
            error_msg = st.session_state.get("component_load_error")
            if error_msg:
                st.error(f"Initialization Error: {error_msg}")
        st.subheader("RAG Settings")
        st.toggle("Hybrid Mode", value=True, key="rec_hybrid_mode", help="Allows AI to use general knowledge alongside retrieved documents.")
        st.toggle("Concise Prompting", value=True, key="rec_concise_mode", help="Uses a more direct prompt for faster, less conversational answers.")
        st.markdown("---")
        st.info("This is a project by Nathan Fane. [View on GitHub](https://github.com/Fane-Nathan/Research-Assistant/tree/alpha-release)")

def main():
    if 'is_initialized' not in st.session_state:
        state_defaults = {'is_initialized': True, 'recommendation_processing': False, 'llm_answer': None, 'context_sources': [], 'evaluation_running': False, 'evaluation_results': None, 'force_component_reload': True}
        for key, value in state_defaults.items():
            if key not in st.session_state: st.session_state[key] = value
    if st.session_state.force_component_reload:
        initialize_and_load_components()

    render_sidebar()
    st.title("Research Paper Assistant")
    
    if st.session_state.get('knowledge_base_is_empty', True):
        st.info("Welcome! Your knowledge base is empty. Please add documents via the 'Update KB' tab.", icon="🎯")
    
    tab_titles = ["🧠 Recommend", "⏬ Update KB", "🔍 Search arXiv", "📊 Evaluation", "⚙️ How It Works"]
    rec_tab, update_tab, search_tab, eval_tab, how_tab = st.tabs(tab_titles)

    with rec_tab:
        st.markdown("##### 💬 Pose Your Research Question")
        query = st.text_area("Enter your question:", height=120, key="rec_query", placeholder="e.g., 'Elucidate the core principles of Retrieval-Augmented Generation.'", label_visibility="collapsed")
        
        is_ready = st.session_state.get('components_loaded_successfully', False)
        if st.button("✨ Generate Response", type="primary", use_container_width=True, disabled=not query or not is_ready):
            st.toast("Synthesizing response... Please wait.", icon="🧠")
            params = {"num_final_results": config.TOP_N_RESULTS, "general_mode": st.session_state.rec_hybrid_mode, "concise_mode": st.session_state.rec_concise_mode, "answer_placeholder": st.empty()}
            asyncio.run(handle_recommendation_submission_async(query, **params))
            
        if st.session_state.llm_answer and not st.session_state.recommendation_processing:
            st.markdown("---")
            st.markdown("##### Synthesized Answer")
            st.markdown(st.session_state.llm_answer)

            if st.session_state.context_sources:
                st.markdown("---")
                st.markdown(f"##### 📚 Retrieved Context ({len(st.session_state.context_sources)} Sources)")
                
                for source in clean_context_list(st.session_state.context_sources):
                    expander_title = format_expander_title(source)
                    with st.expander(expander_title):
                        content = source.get('content', '')
                        st.markdown(f"<small>{(content[:350] + '...') if len(content) > 350 else content}</small>", unsafe_allow_html=True)
                        metadata_parts = []
                        url = source.get('url')
                        if url:
                            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                            display_url = domain_match.group(1) if domain_match else "Source"
                            metadata_parts.append(f'<a href="{url}" target="_blank">🔗 {display_url}</a>')
                        if source.get('page_number'):
                            metadata_parts.append(f"Page: {source.get('page_number')}")
                        if metadata_parts:
                            st.markdown(f'<div class="metadata-line">{" | ".join(metadata_parts)}</div>', unsafe_allow_html=True)

    with update_tab:
        st.markdown("##### ⏬ Update Knowledge Base")
        with st.form("data_fetch_form"):
            st.subheader("📚 Fetch from arXiv")
            arxiv_query = st.text_input("arXiv Query", "large language models")
            num_arxiv = st.number_input("Number of Papers", 1, 50, 5)
            st.subheader("🌐 Crawl a Web URL")
            web_url = st.text_input("Single Web URL", placeholder="https://example.com/research.html")
            if st.form_submit_button("🚀 Ingest and Process Data", type="primary", use_container_width=True):
                with st.spinner("⏳ Ingesting data..."):
                    args = argparse.Namespace(arxiv_query=arxiv_query, num_arxiv=num_arxiv, web_urls=[web_url] if web_url else [])
                    status = asyncio.run(setup_data_and_fetch_cli(args))
                    st.success(f"✅ Data ingestion complete! {status}")
                    st.session_state.force_component_reload = True
                    st.rerun()

    with search_tab:
        st.markdown("##### 🔍 Direct arXiv Search")
        with st.form("arxiv_search_form"):
            query = st.text_input("Search Term", "quantum machine learning")
            num_results = st.number_input("Maximum Results", 1, 25, 5)
            if st.form_submit_button("Search arXiv", use_container_width=True):
                with st.spinner("Searching arXiv..."):
                    results = run_arxiv_search_cli(query, num_results)
                    st.session_state.direct_search_results = results
        if 'direct_search_results' in st.session_state and st.session_state.direct_search_results:
            st.markdown("---")
            for paper in st.session_state.direct_search_results:
                st.markdown(f"**{paper.get('title', 'N/A')}**\n_{', '.join(paper.get('authors', []))}_")
                with st.expander("Abstract"):
                    st.markdown(paper.get('summary', 'N/A'))
                st.divider()

    with eval_tab:
        st.markdown("##### 🧐 RAG System Evaluation")
        eval_ready = st.session_state.get('components_loaded_successfully', False) and not st.session_state.get('knowledge_base_is_empty', True)
        if not eval_ready:
            st.warning("Evaluation requires RAG components to be loaded and the knowledge base to be populated.", icon="⚠️")
        
        if st.button("🚀 Run Full Evaluation", disabled=not eval_ready or st.session_state.get('evaluation_running', False), use_container_width=True):
            run_evaluation_in_app()
            st.rerun()

        results = st.session_state.get('evaluation_results')
        if results:
            if "error" in results:
                st.error(f"Evaluation failed with an error: {results['error']}")
            else:
                st.markdown("---")
                st.subheader("Evaluation Results")
                st.info(f"Evaluation completed in **{results.get('total_time_seconds', 0):.2f} seconds**.", icon="⏱️")
                col1, col2, col3 = st.columns(3)
                col1.metric("F1-Score", f"{results.get('f1_score', 0):.4f}")
                col2.metric("Precision", f"{results.get('precision', 0):.4f}")
                col3.metric("Recall", f"{results.get('recall', 0):.4f}")
                col4, col5, col6 = st.columns(3)
                col4.metric("Hit Rate", f"{results.get('hit_rate', 0):.4f}")
                col5.metric("MRR", f"{results.get('mrr', 0):.4f}")
                col6.metric("nDCG", f"{results.get('ndcg', 0):.4f}")
            
    with how_tab:
        st.markdown("### ⚙️ How the System Operates")
        st.markdown("This assistant employs a **Retrieval-Augmented Generation (RAG)** architecture. Here’s a breakdown:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 1. Ingestion\n- **📥 Data Ingestion**\n- **🧩 Text Chunking**\n- **🔢 Embedding & Indexing**")
        with col2:
            st.markdown("#### 2. Retrieval\n- **🔍 Hybrid Search**\n- **📊 Reranking**\n- **🧠 Context Augmentation**")
        with col3:
            st.markdown("#### 3. Generation\n- **✍️ Answer Synthesis**\n- **💡 Streaming**")

if __name__ == "__main__":
    main()