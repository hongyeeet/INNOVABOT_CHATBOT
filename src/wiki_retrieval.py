# wiki_retrieval.py

from typing import List
import re
import requests
from scripts.build_brochure_index import chunk_text

SEARCH_URL = "https://api.wikimedia.org/core/v1/wikipedia/en/search/page"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def _normalize_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def _compute_overlap_score(query: str, snippet: str) -> float:
    """
    Very simple Jaccard-like overlap between query tokens and snippet tokens.
    Returns 0.0â€“1.0 (higher = more similar).
    """
    q_tokens = set(_normalize_text(query))
    s_tokens = set(_normalize_text(snippet))
    if not q_tokens or not s_tokens:
        return 0.0
    inter = q_tokens & s_tokens
    union = q_tokens | s_tokens
    return len(inter) / len(union)


def _search_wikipedia_title(query: str, min_overlap: float = 0.12) -> str:
    """
    Use the Wikimedia search API to resolve a free-form query into the
    best-matching page key (normalized title).

    Applies a simple relevance filter: if the search result's snippet has
    very low token overlap with the query, treat it as 'no good page'.
    """
    headers = {
        "User-Agent": "EngageProBot/1.0 (student project)",
    }
    params = {
        "q": query,
        "limit": 1,
    }

    try:
        resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=8)
    except Exception as e:
        print("DEBUG wiki search error:", e)
        return ""

    if resp.status_code != 200:
        print(
            "DEBUG wiki search status:",
            resp.status_code,
            "body:",
            resp.text[:200],
        )
        return ""

    data = resp.json()
    pages = data.get("pages", [])
    if not pages:
        print("DEBUG wiki search: no pages for query:", query)
        return ""

    best = pages[0]
    title_key = best.get("key") or best.get("title") or ""
    snippet = best.get("excerpt") or best.get("description") or ""

    score = _compute_overlap_score(query, snippet)
    print(f"DEBUG wiki search best key: {title_key}, overlap score: {score:.3f}")

    # If overlap is too low, ignore this result
    if score < min_overlap:
        print("DEBUG wiki search: overlap below threshold, treating as no match")
        return ""

    return title_key


def fetch_wikipedia_text(raw_query: str) -> str:
    """
    Fetch a short summary of a topic from Wikipedia using search + summary.
    """
    title_key = _search_wikipedia_title(raw_query)
    if not title_key:
        return ""

    url = WIKI_SUMMARY_URL.format(title_key)
    print("DEBUG wiki URL:", url)

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "EngageProBot/1.0 (student project)"},
            timeout=8,
        )
    except Exception as e:
        print("DEBUG wiki request error:", e)
        return ""

    print("DEBUG wiki status:", resp.status_code)
    if resp.status_code != 200:
        print("DEBUG wiki body:", resp.text[:300])
        return ""

    data = resp.json()
    parts = []
    for key in ["title", "description", "extract"]:
        if key in data and data[key]:
            parts.append(str(data[key]))
    return "\n".join(parts)


def retrieve_wiki_context(query: str, top_k: int = 4) -> List[str]:
    """
    Retrieve up to top_k chunks from the Wikipedia summary for a query.
    """
    text = fetch_wikipedia_text(query)
    print("DEBUG wiki text length:", len(text), "for query:", query)
    if not text:
        return []

    chunks = chunk_text(text, chunk_size=600, overlap=100)
    print("DEBUG wiki chunks:", len(chunks))
    return chunks[:top_k]