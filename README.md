# INNOVABOT – EngagePro RAG Chatbot

INNOVABOT is a security-first RAG chatbot built with Streamlit and GPT-4o-mini that answers questions about EngagePro’s products and related AI concepts using hybrid FAISS+BM25 brochure search and Wikipedia retrieval.[file:3][file:6][file:16]

## Features

- **Security-first guardrails**: Regex-based PII detection for NRIC, credit cards, emails, and phone numbers plus OpenAI Moderation API content safety checks before any LLM or retrieval calls.[file:3][file:10][file:16]
- **Combined intent & routing classifier**: Single GPT-4o-mini call classifies greeting/statement/question and routes questions to brochure or Wikipedia, using conversation history for context-aware follow-ups.[file:8][file:16]
- **Hybrid brochure RAG**: Adaptive FAISS + BM25 retrieval over the EngagePro brochure with query-aware fusion weights (keyword-heavy vs conceptual vs mixed) for 82.1% retrieval accuracy in tests.[file:3][file:6][file:16]
- **Wikipedia RAG**: Dedicated WikiResearcher agent using Wikipedia API to handle general knowledge and technical questions outside the brochure.[file:3][file:14][file:16]
- **Defensive response generation**: Product completeness validation, confidence scoring, low-confidence wrapping, and hallucination checks to prioritize accurate, well-calibrated answers over fluent but unsupported text.[file:7][file:9][file:13][file:16]
- **Rich evaluation suite**: `evaluation_final.py` runs 120 test cases across retrieval quality, routing, PII guardrails, hallucination prevention, edge cases, and multi-turn conversations.[file:1][file:3][file:16]

## Project Structure

Key files and modules:[file:3]

- `main.py` – Streamlit chat UI, sidebar controls (mode, temperature, show sources, theme), and end-to-end request orchestration.
- `src/agents.py` – Intent and routing logic (`classify_intent_and_route`, `route_query`, `CompanyExpert`, `WikiResearcher`).
- `src/brochure_retrieval_faiss.py` – Hybrid FAISS + BM25 retrieval over pre-built brochure indexes.
- `src/wiki_retrieval.py` – Wikipedia retrieval for general knowledge queries.
- `src/llm_client.py` – OpenAI GPT-4o-mini wrapper with prompt construction, validation, and confidence scoring.
- `src/guardrails.py` – PII detection + content safety filtering (OpenAI Moderation).
- `src/validators.py` – Product completeness checks and enrichment.
- `src/confidence_scorer.py` – Confidence estimation and low-confidence wrapping.
- `src/query_expander.py` – LLM-based query expansion for short/ambiguous queries.
- `scripts/build_brochure_index.py` – One-time script to build FAISS/BM25 indexes from `CompanyBrochure.pdf`.
- `evaluation_final.py` – Comprehensive evaluation harness and report generator.

See `ProjectStructure.md` and the technical report for a deeper architecture overview.[file:3][file:16]

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (Optional) Rebuild brochure indexes if they are missing/corrupted
python build_brochure_index.py


# INNOVABOT Chatbot – Setup Guide


```bash
cd INNOVABOT_PROJECT

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# or Command Prompt
.venv\Scripts\activate.bat


pip install --upgrade pip
pip install -r requirements.txt


#DO NOT RUN THIS (ALERADY RAN) - only run if brochure_bm25.pkl , brochure_chunks.pkl or brochure_faiss.index are deleted or corrupted 
python build_brochure_index.py 

# open .env file and enter open AI api key
OPENAI_API_KEY=your_openai_key_here

# Run in terminal 
streamlit run main.py
