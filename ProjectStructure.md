# INNOVABOT Project Structure

## Root Files
- **`.env`**  
  Stores API keys (OpenAI) for secure credential management, preventing hardcoded secrets in source code.

- **`ASG.pdf`**  
  Assignment brief and requirements document defining project scope, objectives, and evaluation criteria.

- **`evaluation_final.py`**  
  Comprehensive test suite with 120+ test cases across 7 categories (retrieval quality, content accuracy, routing, PII detection, hallucination prevention, edge cases, multi-turn conversations).

- **`main.py`**  
  Streamlit UI entry point coordinating user interactions, message history, and chatbot responses with configurable settings (mode, temperature, source display).

- **`README.md`**  
  Project documentation with setup instructions and usage guidelines.

- **`requirements.txt`**  
  Python dependencies list for reproducible environment setup including Streamlit, OpenAI, FAISS, Sentence-Transformers, and other libraries.

---

## `/data` - Knowledge Base & Indexes

- **`brochure_bm25.pkl`**  
  Lexical (keyword-based) search index using BM25 algorithm for exact term matching and acronym retrieval, complementing semantic search.

- **`brochure_chunks.pkl`**  
  Pre-processed text chunks from brochure with metadata (chunk IDs, source pages, character counts) enabling efficient retrieval and source attribution.

- **`brochure_faiss.index`**  
  Semantic vector embeddings generated with `all-MiniLM-L6-v2` model for similarity-based retrieval using FAISS (Facebook AI Similarity Search).

- **`Company_Brochure.pdf`**  
  Source document containing EngagePro product information (InnovaBot, CX Transformer, AI Engagement Lab), company mission, and technical capabilities.

---

## `/results` - Evaluation Outputs

- **`evaluation_results_*.txt`**  
  Timestamped test reports tracking system performance over development iterations, capturing pass/fail rates, retrieval quality metrics, and identified issues for continuous improvement.

---

## `/scripts` - Data Preprocessing

- **`build_brochure_index.py`**  
  Converts PDF to text chunks using PyPDF2, generates BM25 lexical index and FAISS semantic embeddings for adaptive hybrid retrieval, executed once during setup.

---

## `/src` - Core System Components

### **Routing & Orchestration**

- **`agents.py`**  
  Intent classification and routing logic determining response type (brochure/wiki/greeting/statement) using LLM-based semantic analysis with conversation history context for multi-turn coherence.

---

### **Retrieval Engines**

- **`brochure_retrieval_faiss.py`**  
  Adaptive hybrid retrieval combining semantic (FAISS cosine similarity) and lexical (BM25 keyword matching) search with dynamic per-query weight optimization based on query characteristics (length, specificity, entity presence).

- **`wiki_retrieval.py`**  
  Wikipedia API integration for general knowledge queries using Tavily search API, extracting relevant passages and providing external source citations for non-brochure topics.

---

### **LLM & Response Generation**

- **`llm_client.py`**  
  OpenAI API wrapper managing GPT-4 calls with confidence scoring, token usage tracking, and low-confidence response wrapping to acknowledge uncertainty when context is insufficient.

- **`prompts.py`**  
  System prompts for brochure mode (cite sources, stay factual), wiki mode (explain concepts), and default mode (acknowledge limitations), with PII refusal and unsafe content templates.

---

### **Safety & Quality**

- **`guardrails.py`**  
  PII detection using regex patterns (Singapore NRIC, credit cards, emails, phone numbers) and content safety filtering via OpenAI Moderation API, blocking sensitive data before LLM processing.

- **`validators.py`**  
  Product completeness validation ensuring all 3 EngagePro products (InnovaBot, CX Transformer, AI Engagement Lab) are mentioned in service-overview responses to prevent incomplete answers.

- **`confidence_scorer.py`**  
  Response quality assessment analyzing retrieval scores, context count, and answer completeness to flag uncertain/incomplete answers with caveats ("I don't have sufficient information...").

---

### **Query Enhancement**

- **`query_expander.py`**  
  Query reformulation adding synonyms and domain-specific terminology (e.g., "chatbot" → "InnovaBot", "customer service" → "CX Transformer") to improve retrieval recall for ambiguous or informal queries.

---

**Architecture Summary:**  
6 root files · 4 data artifacts · 11 evaluation reports · 1 build script · 9 core modules forming a modular, production-ready RAG chatbot with security-first design, adaptive retrieval, and multi-layer quality validation.
