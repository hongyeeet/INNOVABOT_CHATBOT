"""
agents.py
Agent definitions for INNOVABOT routing.
- CompanyExpert: answers brochure-based queries
- WikiResearcher: answers general knowledge queries
- classify_intent_and_route: combined intent + routing classifier
- route_query: router that determines which agent to use
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.brochure_retrieval_faiss import retrieve_brochure_context
from src.wiki_retrieval import retrieve_wiki_context
from src.guardrails import contains_pii, sanitize_input
from src.llm_client import generate_response as llm_generate
from openai import OpenAI

# Load environment variables from .env (including OPENAI_API_KEY)
load_dotenv()

# ============================================================================
# Combined Intent + Route Classifier (NEW!)
# ============================================================================

# Dedicated OpenAI client instance for the classifier, using API key from env
_classifier_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_intent_and_route(query: str, history: List[Dict[str, str]] = None) -> str:
    """
    Combined classifier for intent AND routing in one LLM call.

    This function uses GPT-4o-mini to decide both:
    - whether the user input is a greeting / statement / question
    - and, for questions, whether it should go to brochure or wiki.

    Args:
        query: User query to classify
        history: Conversation history for context

    Returns:
        One of: 'greeting', 'statement', 'question_brochure', 'question_wiki'
    """
    if history is None:
        history = []

    # System prompt defines the 4 allowed labels and how to pick between them.
    # It also encodes EngagePro-specific context and rules for follow-ups.
    system_prompt = """You are a combined intent and routing classifier for EngagePro's chatbot.

COMPANY CONTEXT:
EngagePro is a Singapore-based AI customer engagement company at International Business Park.
Products: InnovaBot (knowledge management), CX Transformer (customer service automation), AI Engagement Lab (R&D initiative)

CLASSIFICATION OPTIONS (reply with ONE word only):

1. "greeting" - Empty queries, hello, hi, good morning, nice to meet you
   Examples: "", "hello", "hi there", "good morning"

2. "statement" - User acknowledgments, feedback, gratitude (NOT questions)
   Examples: "thank you", "that's helpful", "I see", "okay", "got it"

3. "question_brochure" - Questions about EngagePro (company, products, services, team, location, contact, initiatives)
   Examples: "What is InnovaBot?", "Where is EngagePro?", "Tell me about CX Transformer", "What does AI Engagement Lab do?"

4. "question_wiki" - Questions about general knowledge or technical concepts NOT specifically about EngagePro
   Examples: "What is artificial intelligence?", "Explain diffusion models", "What is NLP?", "How do transformers work?"

IMPORTANT CONTEXT RULES:
- Consider conversation history for follow-ups
- "any others?" after asking about products → "question_brochure"
- "are they working on this?" → "question_brochure"
- Even if phrased generally, if it refers to EngagePro's initiatives → "question_brochure"

Reply with ONLY ONE word: greeting, statement, question_brochure, or question_wiki"""

    # Build messages list for the chat completion call
    messages = [{"role": "system", "content": system_prompt}]

    # Add last up to 6 turns of conversation history for context-aware routing.
    # Long messages are truncated to keep token usage low and focus on recent info.
    for h in history[-6:]:
        messages.append({
            "role": h["role"],
            "content": h["content"][:250]  # Truncate long messages
        })

    # Add current query to be classified
    messages.append({
        "role": "user",
        "content": f"Classify: {query}"
    })

    try:
        # Single, deterministic classification call
        resp = _classifier_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,  # zero temperature for repeatable outputs
            max_tokens=10,    # tiny budget since we expect a single word
        )

        # Extract and normalize model output
        result = resp.choices[0].message.content.strip().lower()

        valid_results = ["greeting", "statement", "question_brochure", "question_wiki"]
        if result in valid_results:
            print(f"DEBUG Combined classifier: '{query}' -> {result}")
            return result

        # Fallback: if the LLM returns something unexpected, treat as brochure query.
        # Safer for a company FAQ-style chatbot.
        print(f"DEBUG: Unclear response '{result}', defaulting to question_brochure")
        return "question_brochure"

    except Exception as e:
        # Robustness: in case of API failure, default to brochure as safe behavior.
        print(f"DEBUG Classifier error: {e}, defaulting to question_brochure")
        return "question_brochure"


# ============================================================================
# Legacy Classifiers (Keep for backward compatibility, but not used)
# ============================================================================

def classify_with_llm(query: str, history: List[Dict[str, str]] = None) -> str:
    """
    LEGACY: LLM-based classification for route only (brochure vs wiki).

    This earlier classifier only decides 'brochure' or 'wiki'.
    New code should use classify_intent_and_route(), which also
    distinguishes greetings/statements from questions.
    """
    if history is None:
        history = []

    # System prompt: older, simpler routing model (no greeting/statement).
    system_prompt = """You are an intent classifier for EngagePro's customer service chatbot.

COMPANY CONTEXT:
EngagePro is a Singapore-based AI customer engagement company located at International Business Park.
They build AI solutions to revolutionize customer service:
- InnovaBot: AI knowledge management system
- CX Transformer: Customer service automation platform
- AI Engagement Lab: Current focus on productivity and customer engagement tools

CLASSIFICATION RULES:
- "brochure": ANY question about EngagePro (company, products, services, team, location, contact, mission, vision, future plans, current initiatives)
- "wiki": General knowledge, technical concepts, or topics NOT about EngagePro

IMPORTANT CONTEXT:
- Consider the conversation history for follow-ups
- "any others?" after asking about products = still "brochure"
- "are they working on this?" = still "brochure"
- Even if phrased as general knowledge, if it refers to EngagePro's initiatives = "brochure"

EXAMPLES:
- "What is EngagePro?" → brochure
- "What is InnovaBot?" → brochure
- "What is AI Engagement Lab?" → brochure
- "Any other products?" → brochure (if previous was about EngagePro)
- "What is artificial intelligence?" → wiki
- "Explain diffusion models" → wiki
- "What is a transformer?" → wiki (general term, unless asking about CX Transformer)

Reply with ONLY the single word: brochure or wiki"""

    # Build messages with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Add up to last 6 messages of history for context-aware classification
    for h in history[-6:]:
        messages.append({
            "role": h["role"],
            "content": h["content"][:250]  # Truncate very long messages
        })

    # Current query to classify
    messages.append({
        "role": "user",
        "content": f"Classify this query: {query}"
    })

    try:
        # Deterministic classification for brochure/wiki
        resp = _classifier_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,  # Deterministic
            max_tokens=10,
        )

        result = resp.choices[0].message.content.strip().lower()

        if result in ["brochure", "wiki"]:
            print(f"DEBUG LLM classifier: '{query}' -> {result}")
            return result

        # If LLM returns something unexpected, default to brochure for safety.
        print(f"DEBUG: LLM returned unclear response: '{result}', defaulting to brochure")
        return "brochure"

    except Exception as e:
        # On error, default to brochure so the chatbot still works with company info.
        print(f"DEBUG LLM classifier error: {e}, defaulting to brochure")
        return "brochure"  # Safe default


def classify_query(query: str, history: List[Dict[str, str]] = None) -> str:
    """
    LEGACY wrapper: classify a query as 'brochure' or 'wiki'.

    This exists for backward compatibility; new code should call
    classify_intent_and_route() instead.
    """
    return classify_with_llm(query, history)


# ============================================================================
# Agent Classes
# ============================================================================

class CompanyExpert:
    """Agent that answers questions about EngagePro from the brochure.

    This agent runs hybrid FAISS+BM25 retrieval under the hood (via
    retrieve_brochure_context) and returns relevant chunks plus source info.
    """

    @staticmethod
    def answer(query: str) -> Dict[str, Any]:
        """
        Answer a query using EngagePro brochure context.

        Args:
            query: User query

        Returns:
            Dict with:
            - contexts: list of relevant context chunks (strings)
            - source: "EngagePro brochure"
        """
        # Run adaptive hybrid retrieval with score thresholding
        contexts = retrieve_brochure_context(query, top_k=6, min_score=0.3)

        # If no contexts are returned at all, treat as unanswerable from brochure
        if len(contexts) == 0:
            # No matches at all
            return {
                "contexts": [],
                "source": "EngagePro brochure"
            }

        # contexts is a list of (text, score) tuples
        # Look at the best score to detect very weak matches
        best_score = contexts[0][1] if contexts else 0.0
        if best_score < 0.55:  # Threshold for low relevance
            # Very low relevance - likely an unanswerable question;
            # only give the single best chunk to minimize hallucinations.
            context_texts = [contexts[0][0]] if contexts else []
            return {
                "contexts": context_texts,
                "source": "EngagePro brochure"
            }

        # For good matches, keep only the text part of each (text, score) tuple
        context_texts = [c[0] for c in contexts]

        return {
            "contexts": context_texts,
            "source": "EngagePro brochure"
        }


class WikiResearcher:
    """Agent that answers general knowledge questions using Wikipedia.

    This agent delegates retrieval to retrieve_wiki_context and returns
    chunks plus a 'Wikipedia' source tag.
    """

    @staticmethod
    def answer(query: str) -> Dict[str, Any]:
        """
        Answer a query using Wikipedia.

        Args:
            query: User query

        Returns:
            Dict with:
            - contexts: list of Wikipedia chunks (strings or dicts, depending on implementation)
            - source: "Wikipedia"
        """
        # Simple top-k retrieval from Wikipedia API-based retriever
        contexts = retrieve_wiki_context(query, top_k=4)

        return {
            "contexts": contexts,
            "source": "Wikipedia"
        }


# Initialize singleton instances of agents used by the router
company_expert = CompanyExpert()
wiki_researcher = WikiResearcher()


# ============================================================================
# Main Routing Function
# ============================================================================

def route_query(
    query: str,
    mode: str,
    history: List[Dict[str, str]] = None
) -> Dict:
    """
    Route a query to the appropriate agent(s) based on mode.
    PII queries are blocked before any retrieval.

    This function is the main entry point used by the UI:
    it enforces guardrails, chooses the agent, and returns
    contexts plus metadata for downstream LLM generation.

    Args:
        query: Original query from user
        mode: Routing mode - "Brochure only", "Wikipedia", or "Auto"
        history: Conversation history for context-aware routing

    Returns:
        Dict with:
        - contexts: list of context chunks (empty if PII blocked)
        - sources_used: list of sources used (e.g., ["EngagePro brochure"])
        - kind: "brochure", "wiki", or "default"
        - pii_blocked: True if PII was detected and blocked
        - sanitized_query: The query with PII masked (for LLM)
    """
    if history is None:
        history = []

    # Initialize containers for retrieval results
    contexts: List[str] = []
    sources_used: List[str] = []
    pii_blocked = False

    # ✅ STEP 1: Check PII on ORIGINAL query BEFORE anything else
    # If PII is detected, we block and do NOT run retrieval or LLM.
    if contains_pii(query):
        pii_blocked = True
        return {
            "contexts": [],
            "sources_used": [],
            "kind": "default",
            "pii_blocked": True,
            "sanitized_query": query,  # Return original since we're blocking
        }

    # ✅ STEP 2: Sanitize query (mask any remaining PII patterns)
    # This ensures even borderline PII patterns are masked before reaching LLM.
    sanitized_query, had_pii = sanitize_input(query)

    # Use sanitized query for all retrieval and classification from here
    query_for_processing = sanitized_query

    # Route based on explicit mode selection in the UI sidebar
    if mode == "Brochure only":
        # Force brochure retrieval regardless of classifier
        result = company_expert.answer(query_for_processing)
        contexts.extend(result["contexts"])
        sources_used.append(result["source"])
        kind = "brochure"

    elif mode == "Wikipedia":
        # Force Wikipedia retrieval regardless of classifier
        result = wiki_researcher.answer(query_for_processing)
        contexts.extend(result["contexts"])
        sources_used.append(result["source"])
        kind = "wiki"

    else:  # mode == "Auto"
        # Auto mode uses the legacy brochure/wiki classifier for backward compatibility.
        # (Could be upgraded to use classify_intent_and_route in future.)
        cls = classify_query(query_for_processing, history)

        if cls == "brochure":
            result = company_expert.answer(query_for_processing)
        else:
            result = wiki_researcher.answer(query_for_processing)

        contexts.extend(result["contexts"])
        sources_used.append(result["source"])
        kind = cls

    return {
        "contexts": contexts,
        "sources_used": sources_used,
        "kind": kind,
        "pii_blocked": False,
        "sanitized_query": sanitized_query,  # Return sanitized query for LLM
    }
