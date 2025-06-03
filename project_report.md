# Final Project Report: StudyAssistant - A Machine Learning-Powered Research Assistant

**Group Members:** [Insert Name 1], [Insert Name 2], [Insert Name 3]
**Date:** June 2, 2025

## 1. Background & Problem

### 1.1. Introduction
The rapid growth of academic literature presents a significant challenge for researchers, students, and professionals. Sifting through vast numbers of papers to find relevant information, understand complex topics, and synthesize knowledge is a time-consuming and often overwhelming task. Traditional keyword-based search engines can struggle with semantic understanding, leading to noisy or incomplete results. This project, "StudyAssistant," aims to address these challenges by developing a Machine Learning-based application that assists users in navigating and understanding academic content more efficiently.

### 1.2. Problem Statement
The core problem is information overload and the difficulty in efficiently accessing and synthesizing specific knowledge from a large corpus of academic documents. Researchers need tools that can:
*   Understand the semantic meaning of their queries, not just keywords.
*   Retrieve highly relevant documents and specific passages within them.
*   Provide concise summaries or answers based on the retrieved information.
*   Facilitate a deeper understanding of complex topics by connecting related concepts from various sources.

### 1.3. Chosen Theme: Recommendation & Information Retrieval
This project falls under the theme of **Recommendation Systems** and advanced **Information Retrieval**. Specifically, it focuses on recommending relevant academic papers and generating context-aware answers to user queries, leveraging a Retrieval Augmented Generation (RAG) approach. The "StudyAssistant" acts as an intelligent system that recommends information and aids in its comprehension.

## 2. Approach & Methodology

### 2.1. System Architecture: Retrieval Augmented Generation (RAG)
The StudyAssistant employs a Retrieval Augmented Generation (RAG) architecture. This approach combines the strengths of information retrieval systems (to find relevant context) and large language models (LLMs) (to generate human-like responses).

The main components are:
1.  **Data Ingestion and Preprocessing Pipeline:** Collects, cleans, and prepares academic papers for retrieval.
2.  **Retriever:** Identifies and ranks relevant document chunks based on a user's query. This project utilizes a hybrid approach.
3.  **Generator:** An LLM that takes the user's query and the retrieved context to generate a comprehensive answer.
4.  **User Interface:** A web-based application allowing users to input queries and view results.

*(Self-reflection for "Report: Learning Outcome & Proficiency - Comprehension")*:
*Machine learning concepts employed here include natural language processing (NLP) for text understanding, embedding models for semantic representation, similarity search algorithms for retrieval, and generative models for answer synthesis. A key challenge is ensuring the relevance and faithfulness of the generated answers to the source material, and managing the trade-off between retrieval breadth and LLM context window limitations.*

### 2.2. Dataset
*   **Source:** The primary dataset consists of academic papers, primarily sourced from the arXiv repository. The system is designed to be extensible to other sources like local PDF collections or web URLs.
*   **Collection:**
    *   arXiv papers are fetched using the arXiv API (as seen in `resource_fetcher.py`).
    *   A continuous fetching mechanism (`continuous_fetch_with_dedup.py`) is implemented to keep the dataset updated and avoid duplicates using hash-based deduplication (`document_hashes.json`).
*   **Volume & Characteristics:** [Specify approximate number of papers, subject domains, e.g., Computer Science, Physics, etc. Mention if specific fields were targeted, e.g., `astro-ph`, `cs.AI`]. The data is primarily in PDF and sometimes HTML format, containing text, figures, and equations.

### 2.3. Preprocessing Methods
Raw data undergoes several preprocessing steps managed by `hybrid_search_rag/data_handling/data_manager.py` and `hybrid_search_rag/text_processing/`:
1.  **Text Extraction:**
    *   PDFs: Content extracted using libraries like `PyMuPDF`.
    *   Web URLs/HTML: Text extracted using `trafilatura` and `BeautifulSoup` (as referenced in `resource_fetcher.py`).
2.  **Text Cleaning:** Removal of irrelevant characters, normalization of whitespace, and potentially more advanced cleaning (e.g., handling ligatures, broken words from PDF extraction). The `_clean_text` function in `resource_fetcher.py` is an example.
3.  **Chunking:** Documents are segmented into smaller, manageable chunks (e.g., paragraphs or fixed-size overlapping segments). This is crucial for effective retrieval and fitting context into the LLM's prompt.
4.  **Embedding Generation:**
    *   Each text chunk is converted into a dense vector representation (embedding) using a pre-trained sentence transformer model (e.g., via `hybrid_search_rag/embedding_services/gemini_embedder.py` or similar). These embeddings capture the semantic meaning of the text.
    *   Embeddings are stored (e.g., in `combined_embeddings.npy`) for efficient similarity search.
5.  **Metadata Storage:** Metadata for each document (title, authors, source URL, publication date, abstract, extracted text path, etc.) is stored in `combined_metadata.json`.

### 2.4. Algorithm Selection
1.  **Retriever (`hybrid_search_rag/retrieval_algorithm/hybrid_recommender.py`):**
    *   **Sparse Retrieval (Keyword-based):** BM25 (Okapi BM25) is used for efficient keyword matching. The BM25 index is pre-computed and stored (e.g., `bm25_index.pkl`). This helps find documents with exact keyword matches.
    *   **Dense Retrieval (Semantic Search):** Cosine similarity is used to compare the query embedding with the pre-computed chunk embeddings. This finds semantically similar content even if keywords don't match exactly.
    *   **Hybrid Fusion:** Results from sparse and dense retrieval are combined using a fusion technique (e.g., Reciprocal Rank Fusion - RRF, or weighted scoring) to leverage the strengths of both methods.
2.  **Generator (`hybrid_search_rag/llm_services/llm_interface.py`):**
    *   A pre-trained Large Language Model (LLM) (e.g., from OpenAI, Google Gemini, or a locally hosted model) is used.
    *   **Prompt Engineering:** The LLM is prompted with the user's query and the top-k retrieved text chunks. The prompt is carefully designed to instruct the LLM to synthesize an answer based *only* on the provided context, citing sources if possible.
    *   [Mention specific LLM used, e.g., GPT-3.5-turbo, Gemini-Pro, etc.]

*(Self-reflection for "Application: Learning Outcome & Proficiency - Analysis experiment classification and clustering algorithm")*:
*While the primary task is RAG, an implicit **classification** occurs during retrieval: documents are classified as "relevant" or "not relevant" to the query. The semantic embeddings used for dense retrieval inherently perform a form of **clustering** in the vector space, where similar documents/chunks are grouped closer together. The hybrid recommender then selects candidates from these implicit clusters. Explicit clustering algorithms (e.g., k-means on embeddings) could be used for topic modeling or diversifying results, but are not the core of the current RAG pipeline. The effectiveness of retrieval can be seen as a proxy for how well the system "classifies" and "clusters" information in response to a query.*

## 3. Implementation

### 3.1. Technologies Used
*   **Programming Language:** Python (version 3.10+)
*   **Core ML/NLP Libraries:**
    *   `scikit-learn`: For BM25 and other potential ML tasks.
    *   `nltk`: For text processing (tokenization, sentence segmentation - see `nltk_data/`).
    *   `transformers` (Hugging Face): For accessing embedding models (if applicable).
    *   `sentence-transformers`: For generating text embeddings.
    *   Specific LLM SDKs: e.g., `openai`, `google-generativeai`.
*   **Data Handling & Storage:**
    *   `numpy`: For numerical operations, especially with embeddings.
    *   `pandas`: For data manipulation (if used).
    *   `faiss` or `annoy`: For efficient similarity search in vector spaces (if implemented, otherwise numpy-based search).
    *   JSON: For metadata storage (`combined_metadata.json`).
    *   Pickle: For storing Python objects like the BM25 index (`bm25_index.pkl`).
*   **Web Framework & UI:**
    *   `Streamlit`: For creating the interactive web application interface (`app.py`, `requirements_streamlit.txt`).
*   **Asynchronous Operations & Web Fetching:**
    *   `asyncio`, `aiohttp`: For efficient, non-blocking fetching of web resources and arXiv data.
    *   `Playwright`: For more complex web scraping tasks that might require JavaScript rendering (as seen in `resource_fetcher.py`).
*   **Configuration Management:** `hybrid_search_rag/config.py` for managing API keys and parameters.
*   **Logging:** `logging` module for application monitoring and debugging (e.g., `app.log`).
*   **Dependency Management:** `requirements.txt`, `requirements_development.txt`, `requirements_streamlit.txt`.

### 3.2. Application Features (`app.py`)
The application provides a user-friendly interface with the following main features:
1.  **Data Input:**
    *   A text input field where users can type their research questions or topics of interest.
    *   [Optional: Ability to upload documents for querying, or specify arXiv categories/keywords for initial data fetching if implemented in the UI].
2.  **Processing (RAG Pipeline):**
    *   Upon query submission, the backend processes the query:
        *   The query is embedded.
        *   The hybrid retriever fetches the most relevant text chunks from the indexed academic papers.
        *   The LLM receives the query and retrieved context.
3.  **Informative Output:**
    *   The LLM-generated answer/summary is displayed clearly.
    *   Sources: Relevant document titles, authors, and snippets from which the answer was derived are presented, often with links to the original papers. This allows users to verify information and delve deeper.
    *   [Optional: Confidence scores, visualization of semantic similarity, or related topics if implemented].

## 4. Evaluation & Results

### 4.1. Model Testing & Evaluation Framework
The system's performance was evaluated using a combination of automated metrics and qualitative human assessment. The `hybrid_search_rag/evaluation/` directory contains scripts and datasets for this purpose.
*   **Dataset for Evaluation:**
    *   A curated set of question-answer pairs relevant to the corpus was created (e.g., `evaluation_dataset_llm_labeled.json`).
    *   Ground truth relevant documents/passages were identified for each question.
*   **Retrieval Evaluation:**
    *   **Metrics:**
        *   **Mean Reciprocal Rank (MRR):** Measures the rank of the first relevant document.
        *   **Normalized Discounted Cumulative Gain (nDCG@k):** Evaluates the quality of ranking for the top-k retrieved documents.
        *   **Precision@k:** Proportion of relevant documents among the top-k retrieved.
        *   **Recall@k:** Proportion of all relevant documents that are retrieved in the top-k.
    *   **Methodology:** For each query in the evaluation set, the retriever component was run, and the ranked list of retrieved chunks was compared against the ground truth.
*   **End-to-End RAG Evaluation (Generation Quality):**
    *   **Metrics (if automated):**
        *   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Compares the generated answer to a reference summary.
        *   **BERTScore:** Uses contextual embeddings to compare the semantic similarity between the generated answer and reference.
        *   **Faithfulness/Attribution:** Metrics to assess if the generated answer is factually consistent with and attributable to the provided source documents. (e.g., using an LLM-as-a-judge approach or custom scripts).
    *   **Human Evaluation:**
        *   A subset of generated answers was reviewed by humans for:
            *   **Relevance:** Is the answer relevant to the query?
            *   **Coherence & Readability:** Is the answer well-structured and easy to understand?
            *   **Faithfulness:** Does the answer accurately reflect the information in the source documents?
            *   **Completeness:** Does the answer adequately address the query based on the available context?
        *   The `interactive_evaluation_labeler.py` script might have been used to facilitate this.

### 4.2. Results
[Present quantitative results for retrieval and generation metrics. Use tables or charts if possible.]
*   **Retrieval Performance:**
    *   MRR: [Value]
    *   nDCG@10: [Value]
    *   Precision@10: [Value]
    *   Recall@10: [Value]
*   **Generation Quality:**
    *   ROUGE-L: [Value] (if applicable)
    *   BERTScore F1: [Value] (if applicable)
    *   Human Evaluation Scores (e.g., average scores on a Likert scale for relevance, faithfulness): [Describe findings]

**Example:**
"The hybrid retrieval system achieved an MRR of 0.75 and nDCG@10 of 0.82, indicating strong performance in ranking relevant documents highly. Human evaluation of generated answers showed an average faithfulness score of 4.2/5.0."

### 4.3. Challenges Faced & Solutions
*   **Challenge 1: PDF Parsing Quality:** Extracting clean text from diverse PDF layouts was difficult, leading to noisy input for embedding and retrieval.
    *   **Solution:** Experimented with multiple PDF parsing libraries (e.g., PyMuPDF, Tika). Implemented more robust post-processing cleaning steps. For very complex layouts, some manual intervention or more advanced layout-aware models might be needed in the future.
*   **Challenge 2: Optimizing Hybrid Search Parameters:** Finding the right balance and fusion strategy for BM25 and semantic search results required experimentation.
    *   **Solution:** Conducted ablation studies by varying weights for sparse and dense components and testing different fusion methods (e.g., simple weighted sum vs. RRF) on the evaluation dataset.
*   **Challenge 3: LLM Hallucination & Faithfulness:** Ensuring the LLM generates answers strictly based on provided context and avoids making up information.
    *   **Solution:** Implemented strict prompting techniques (e.g., "Answer based *only* on the provided documents..."). Explored using smaller, more focused LLMs or fine-tuning for the summarization/Q&A task. Implemented source citation to allow users to verify.
*   **Challenge 4: Scalability of Embedding Storage & Search:** As the number of documents grows, naive similarity search becomes slow.
    *   **Solution:** [If implemented: Used approximate nearest neighbor (ANN) search libraries like FAISS or Annoy for faster retrieval. Otherwise: This is a known area for future improvement for larger datasets.]
*   **Challenge 5: Managing API Rate Limits and Costs:** Frequent calls to external LLM and embedding APIs can be costly and subject to rate limits.
    *   **Solution:** Implemented caching for embeddings and LLM responses for identical queries/contexts. Used `asyncio.Semaphore` for rate limiting external API calls (`resource_fetcher.py`). Optimized batching where possible.

## 5. Group Task Distribution

| Member Name      | Tasks Assigned                                                                                                | Contribution Highlights                                                                                                |
|------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [Student Name 1] | e.g., Lead on Data Ingestion & Preprocessing, arXiv API integration, `resource_fetcher.py` development.       | e.g., Successfully implemented robust asynchronous data fetching pipeline; designed deduplication strategy.          |
| [Student Name 2] | e.g., Lead on Retrieval Algorithms, BM25 & Semantic Search implementation, `hybrid_recommender.py` development. | e.g., Optimized hybrid search fusion for improved relevance; developed evaluation scripts for retrieval metrics.    |
| [Student Name 3] | e.g., Lead on LLM Integration & UI Development, `llm_interface.py` and `app.py` (Streamlit) development.       | e.g., Designed effective prompts for faithful generation; created an intuitive user interface for query and results. |
| **All Members**  | Documentation, Testing, Debugging, Report Writing, Project Management.                                        | e.g., Collaborated effectively on integration, troubleshooting, and ensuring project milestones were met.            |

*(This section needs to be filled by the group with actual task distribution)*

## 6. Conclusion & Future Work

### 6.1. Conclusion
The StudyAssistant project successfully demonstrates the application of Machine Learning, specifically Retrieval Augmented Generation, to address the problem of information overload in academic research. The developed application allows users to input natural language queries and receive contextually relevant, LLM-generated answers sourced from a corpus of academic papers. The hybrid retrieval approach, combining sparse and dense methods, proved effective in surfacing relevant information. The project met its core objectives of creating a functional ML application with data input, processing, and informative output.

### 6.2. Future Work
*   **Enhanced Data Sources:** Integrate more diverse data sources (e.g., other academic databases, textbooks, lecture notes).
*   **Advanced NLP Features:**
    *   **Query Understanding:** Implement query expansion, disambiguation, or intent recognition.
    *   **Conversational Interface:** Allow for follow-up questions and a more interactive dialogue.
*   **Improved Evaluation:** Develop more sophisticated automated metrics for faithfulness and explore real-time user feedback mechanisms.
*   **Scalability & Performance:**
    *   Transition to a dedicated vector database for embeddings (e.g., Pinecone, Weaviate, Milvus).
    *   Optimize LLM inference (e.g., quantization, knowledge distillation if using local models).
*   **Personalization:** Allow users to create profiles and receive recommendations tailored to their research interests.
*   **Fine-tuning:** Fine-tune embedding models and/or LLMs on the specific domain of academic papers for improved performance.
*   **Advanced UI/UX:** Incorporate features like knowledge graphs, citation network visualization, or collaborative annotation.

## 7. Appendix (Optional)

*   Code snippets (if particularly illustrative and not too long).
*   Detailed evaluation results or charts.
*   User manual for the application.

---
*This report structure is based on the provided guidelines and the inferred nature of the "StudyAssistant" project from the workspace structure. Please adapt and fill in the details specific to your group's work.*
