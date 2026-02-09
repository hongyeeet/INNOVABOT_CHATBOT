# llm_client.py

import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from src.prompts import build_system_message, build_user_message, build_pii_refusal_message
from src.guardrails import sanitize_output
from src.validators import (
    validate_product_completeness, 
    enrich_response_with_missing_products,
    check_for_hallucinations
)
from src.confidence_scorer import (  # ✅ NEW IMPORT
    calculate_confidence_score,
    wrap_low_confidence_response,
    generate_confidence_explanation
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_classifier_client = client

MODEL_NAME = "gpt-4o-mini"


def _build_messages(
    kind: str,
    user_query: str,
    contexts: List[str],
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Internal helper to create OpenAI-style messages:
    - system message (task + safety + context)
    - short history
    - current user message
    """
    system_msg = build_system_message(kind, contexts)
    user_msg = build_user_message(user_query)
    
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]
    
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    
    messages.append({"role": "user", "content": user_msg})
    
    return messages


def generate_response(
    kind: str,
    user_query: str,
    contexts: List[str],
    history: List[Dict[str, str]],
    temperature: float = 0.3,
) -> Tuple[str, Dict[str, int]]:
    """
    Unified LLM call with validation and confidence scoring.
    
    ✅ Fix 1: Product completeness validation
    ✅ Fix 2: Confidence scoring and hallucination detection
    
    Args:
        kind: Response type ("brochure", "wiki", or "default")
        user_query: User's question
        contexts: Retrieved context chunks
        history: Conversation history
        temperature: LLM temperature (0-1)
    
    Returns:
        Tuple of (response_text, usage_stats)
    """
    # Build messages
    messages = _build_messages(kind, user_query, contexts, history)
    
    # Call OpenAI API
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )
    
    # Extract response
    raw_text = resp.choices[0].message.content
    safe_text = sanitize_output(raw_text)
    
    # ✅ FIX 1: Validate product completeness
    validation = validate_product_completeness(user_query, safe_text, kind)
    
    if validation["needs_enrichment"]:
        print(f"⚠️  Response incomplete - missing: {validation['missing_products']}")
        safe_text = enrich_response_with_missing_products(
            safe_text, 
            validation["missing_products"]
        )
        print(f"✅ Response enriched with missing products")
    
    # ✅ FIX 2: Score confidence and detect hallucinations
    confidence_data = calculate_confidence_score(safe_text, contexts)
    
    # Log confidence for debugging
    confidence_explanation = generate_confidence_explanation(confidence_data)
    print(f"📊 {confidence_explanation}")
    
    # Wrap low-confidence responses with caveats
    safe_text = wrap_low_confidence_response(safe_text, confidence_data, kind=kind)
    
    # Optional: Additional hallucination check
    hallucination_check = check_for_hallucinations(safe_text, contexts)
    if hallucination_check["has_hallucinations"]:
        print(f"⚠️  Potential hallucinations detected: {hallucination_check['hallucinations']}")
    
    # Collect usage statistics
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
        "was_enriched": validation["needs_enrichment"],
        "missing_products_count": len(validation["missing_products"]),
        # ✅ NEW: Confidence metrics
        "confidence_score": confidence_data["confidence"],
        "reliability_tier": confidence_data["reliability_tier"],
        "has_uncertainty": confidence_data["has_uncertainty"],
        "has_speculation": confidence_data["has_speculation"],
        "context_support": confidence_data["context_support"],
    }
    
    return safe_text, usage
