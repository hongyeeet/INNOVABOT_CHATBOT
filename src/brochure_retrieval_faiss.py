# brochure_retrieval_faiss.py

"""
Hybrid retrieval utilities for EngagePro brochure using FAISS + BM25.

This module LOADS pre-built indexes and performs retrieval.
To BUILD indexes, run: python build_brochure_index.py

✅ ENHANCEMENT: Combines semantic (FAISS) + lexical (BM25) retrieval
"""

from typing import List, Tuple
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# Paths (must match build_brochure_index.py)
BROCHURE_FAISS_INDEX_PATH = "data/brochure_faiss.index"
BROCHURE_CHUNKS_PATH = "data/brochure_chunks.pkl"
BROCHURE_BM25_PATH = "data/brochure_bm25.pkl"

# Global caches
_embedder = None
_brochure_chunks: List[str] = []
_faiss_index = None
_bm25_index = None


def get_embedder() -> SentenceTransformer:
    """Lazily create and return a SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _load_indexes() -> bool:
    """
    Load all pre-built indexes from disk.
    
    Returns:
        True if successful, False otherwise
    """
    global _brochure_chunks, _faiss_index, _bm25_index
    
    # Check if all files exist
    required_files = [
        BROCHURE_CHUNKS_PATH,
        BROCHURE_FAISS_INDEX_PATH,
        BROCHURE_BM25_PATH
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing index files:")
        for f in missing_files:
            print(f"   - {f}")
        print()
        print("Please run: python build_brochure_index.py")
        return False
    
    try:
        # Load chunks
        with open(BROCHURE_CHUNKS_PATH, "rb") as f:
            _brochure_chunks = pickle.load(f)
        
        # Load FAISS index
        _faiss_index = faiss.read_index(BROCHURE_FAISS_INDEX_PATH)
        
        # Load BM25 index
        with open(BROCHURE_BM25_PATH, "rb") as f:
            _bm25_index = pickle.load(f)
        
        print(f"✓ Loaded indexes: {len(_brochure_chunks)} chunks, FAISS + BM25 ready")
        return True
    
    except Exception as e:
        print(f"❌ Error loading indexes: {e}")
        print("Please rebuild indexes: python build_brochure_index.py")
        return False


def _ensure_indexes_loaded():
    """Ensure indexes are loaded before retrieval."""
    global _faiss_index, _bm25_index, _brochure_chunks
    
    # Already loaded
    if _faiss_index is not None and _bm25_index is not None and _brochure_chunks:
        return
    
    # Try to load
    if not _load_indexes():
        raise RuntimeError(
            "Indexes not found. Please run: python build_brochure_index.py"
        )


def retrieve_brochure_context(
    query: str,
    top_k: int = 6,
    min_score: float = 0.2,
    fusion_weight: float = None  # ✅ NEW: Auto-detect if None
) -> List[Tuple[str, float]]:
    """
    Hybrid retrieval combining FAISS (semantic) + BM25 (lexical).
    
    ✅ ADAPTIVE FUSION: Automatically adjusts semantic/lexical balance based on query type.
    
    Query Types:
    - Keyword-heavy (platforms, numbers, names) → 50% semantic, 50% lexical
    - Conceptual (how, why, benefits) → 70% semantic, 30% lexical
    
    Args:
        query: User query
        top_k: Number of chunks to return
        min_score: Minimum similarity threshold (0-1)
        fusion_weight: Weight for semantic scores (auto-detected if None)
            - None = Auto-detect based on query (RECOMMENDED)
            - 0.7 = 70% semantic, 30% lexical (manual override)
            - 0.5 = 50% semantic, 50% lexical (manual override)
    
    Returns:
        List of (chunk_text, fused_score) tuples sorted by relevance
    """
    # Ensure indexes are loaded
    _ensure_indexes_loaded()
    
    # ========== ADAPTIVE FUSION WEIGHT ==========
    if fusion_weight is None:
        fusion_weight = _detect_fusion_weight(query)
    
    # ========== 1. SEMANTIC RETRIEVAL (FAISS) ==========
    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb_normalized = np.array([q_emb], dtype=np.float32)
    faiss.normalize_L2(q_emb_normalized)
    
    # Get candidates
    distances, indices = _faiss_index.search(
        q_emb_normalized, 
        min(top_k * 2, len(_brochure_chunks))
    )
    
    semantic_scores = {}
    for idx, dist in zip(indices[0], distances[0]):
        semantic_scores[int(idx)] = float(dist)
    
    # ========== 2. LEXICAL RETRIEVAL (BM25) ==========
    query_tokens = query.lower().split()
    bm25_scores_raw = _bm25_index.get_scores(query_tokens)
    
    # Normalize BM25 scores to 0-1 range
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
    bm25_scores = {i: score / max_bm25 for i, score in enumerate(bm25_scores_raw)}
    
    # ========== 3. FUSE SCORES ==========
    all_indices = set(semantic_scores.keys()) | set(bm25_scores.keys())
    
    fused_scores = {}
    for idx in all_indices:
        semantic = semantic_scores.get(idx, 0.0)
        lexical = bm25_scores.get(idx, 0.0)
        fused_scores[idx] = (fusion_weight * semantic) + ((1 - fusion_weight) * lexical)
    
    # ========== 4. SORT AND FILTER ==========
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    results: List[Tuple[str, float]] = []
    for idx, score in sorted_results[:top_k]:
        if score < min_score:
            continue
        chunk_text = _brochure_chunks[idx]
        results.append((chunk_text, score))
    
    # Debug logging
    if results:
        print(f"🔍 Hybrid retrieval: {len(results)} chunks (semantic: {fusion_weight:.0%}, lexical: {1-fusion_weight:.0%})")
    
    return results


def _detect_fusion_weight(query: str) -> float:
    """
    Automatically detect optimal fusion weight based on query characteristics.
    
    Strategy:
    - Keyword-heavy queries (product names, numbers, platforms) → More lexical (50/50)
    - Conceptual queries (how, why, benefits, explain) → More semantic (70/30)
    - Mixed queries → Balanced (60/40)
    
    Args:
        query: User query
    
    Returns:
        Optimal fusion weight (0.5-0.7)
    """
    query_lower = query.lower()
    
    # ===== KEYWORD-HEAVY INDICATORS =====
    # Specific product/company names
    specific_names = [
        'innovabot', 'cx transformer', 'ai engagement lab',
        'engagepro', 'slack', 'teams', 'sharepoint', 'confluence',
        'microsoft', 'fortune 500'
    ]
    
    # Numbers and percentages
    has_numbers = any(char.isdigit() for char in query)
    
    # Acronyms
    acronyms = ['nlp', 'nps', 'ai', 'ml', 'api', 'crm']
    
    # Technical terms
    technical_terms = [
        'platform', 'integration', 'api', 'workflow', 
        'analytics', 'dashboard', 'deployment'
    ]
    
    # ===== CONCEPTUAL INDICATORS =====
    conceptual_words = [
        'how', 'why', 'what is', 'explain', 'describe',
        'benefit', 'advantage', 'help', 'improve', 'enhance',
        'difference', 'compare', 'better', 'best'
    ]
    
    # ===== COUNT MATCHES =====
    keyword_score = 0
    conceptual_score = 0
    
    # Check specific names
    if any(name in query_lower for name in specific_names):
        keyword_score += 2
    
    # Check numbers
    if has_numbers:
        keyword_score += 2
    
    # Check acronyms
    if any(acronym in query_lower.split() for acronym in acronyms):
        keyword_score += 1
    
    # Check technical terms
    if any(term in query_lower for term in technical_terms):
        keyword_score += 1
    
    # Check conceptual words
    if any(word in query_lower for word in conceptual_words):
        conceptual_score += 2
    
    # Check question type
    if query.strip().endswith('?'):
        conceptual_score += 1
    
    # ===== DECIDE FUSION WEIGHT =====
    # Heavy keyword query → 50/50 (equal weight)
    if keyword_score >= 3 or (keyword_score >= 2 and conceptual_score == 0):
        weight = 0.5
        strategy = "keyword-heavy"
    
    # Heavy conceptual query → 75/25 (favor semantic)
    elif conceptual_score >= 3 or (conceptual_score >= 2 and keyword_score == 0):
        weight = 0.75
        strategy = "conceptual"
    
    # Mixed query → 60/40 (slight semantic preference)
    else:
        weight = 0.6
        strategy = "balanced"
    
    print(f"🎯 Query type: {strategy} (semantic: {weight:.0%}, lexical: {1-weight:.0%})")
    
    return weight
