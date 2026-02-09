"""
confidence_scorer.py
Detects low-confidence responses that may be hallucinations or guesses
"""

from typing import List, Dict, Tuple
import re

# Phrases that indicate the LLM is uncertain
LOW_CONFIDENCE_PHRASES = [
    r"i'?m not sure",
    r"i don'?t have (specific )?details?",
    r"i don'?t have (specific )?information",
    r"the brochure does not (provide|mention|state|include)",
    r"i cannot provide",
    r"not stated in the brochure",
    r"not mentioned in the brochure",
    r"not available in the brochure",
    r"i'?m unable to",
    r"i don'?t know",
]

# Phrases that suggest the LLM is speculating/guessing
SPECULATIVE_PHRASES = [
    r"might be",
    r"could be",
    r"possibly",
    r"perhaps",
    r"it seems",
    r"appears to",
    r"likely",
    r"probably",
    r"may be",
    r"suggests that",
]

# Phrases that indicate making things up (hallucination red flags)
HALLUCINATION_RED_FLAGS = [
    r"as far as i know",
    r"based on general knowledge",
    r"typically",
    r"generally",
    r"in my understanding",
    r"from what i understand",
]


def calculate_confidence_score(response: str, contexts: List[str]) -> Dict[str, any]:
    """
    Calculate confidence score for a response.
    
    ✅ IMPROVED: Better handling of complete, detailed answers
    """
    response_lower = response.lower()
    
    # 1. Check for explicit uncertainty
    has_uncertainty = any(
        re.search(pattern, response_lower) 
        for pattern in LOW_CONFIDENCE_PHRASES
    )
    
    # 2. Check for speculation
    has_speculation = any(
        re.search(pattern, response_lower) 
        for pattern in SPECULATIVE_PHRASES
    )
    
    # 3. Check for hallucination red flags
    has_hallucination_flags = any(
        re.search(pattern, response_lower) 
        for pattern in HALLUCINATION_RED_FLAGS
    )
    
    # 4. Calculate context support (word overlap)
    if contexts and len(contexts) > 0:
        context_support = _calculate_context_overlap(response, contexts)
    else:
        context_support = 0.0
    
    # ✅ NEW: Boost confidence for responses with product names (indicates grounding)
    has_product_names = any(
        product.lower() in response_lower 
        for product in ["innovabot", "cx transformer", "ai engagement lab"]
    )
    
    # ✅ NEW: Boost for structured responses (numbered lists, bullets)
    has_structure = bool(re.search(r'(\d+\.|•|\-)\s+\*\*', response))
    
    # 5. Calculate final confidence score
    confidence = context_support
    
    # ✅ IMPROVED: Only penalize if ALSO low context support
    if has_uncertainty:
        if context_support < 0.3:
            confidence *= 0.6  # Strong penalty only if really unsupported
        else:
            confidence *= 0.8  # Mild penalty if context supports it
    
    if has_speculation:
        confidence *= 0.75
    
    if has_hallucination_flags:
        confidence *= 0.5
    
    # ✅ NEW: Boost for well-grounded responses
    if has_product_names and not has_uncertainty:
        confidence = min(confidence * 1.3, 1.0)  # Boost if mentions products
    
    if has_structure and not has_uncertainty:
        confidence = min(confidence * 1.2, 1.0)  # Boost for structured answers
    
    # ✅ NEW: Context-rich responses are good (removed word count penalty)
    if context_support > 0.5 and not has_speculation:
        confidence = min(confidence * 1.15, 1.0)
    
    # Determine reliability tier
    if confidence >= 0.65:  # ✅ Lowered from 0.7
        reliability_tier = "high"
    elif confidence >= 0.35:  # ✅ Lowered from 0.4
        reliability_tier = "medium"
    else:
        reliability_tier = "low"
    
    return {
        "confidence": round(confidence, 3),
        "has_uncertainty": has_uncertainty,
        "has_speculation": has_speculation,
        "has_hallucination_flags": has_hallucination_flags,
        "context_support": round(context_support, 3),
        "is_reliable": confidence > 0.35,  # ✅ Lowered threshold
        "reliability_tier": reliability_tier,
    }



def _calculate_context_overlap(response: str, contexts: List[str]) -> float:
    """
    Calculate word overlap between response and retrieved contexts.
    
    Higher overlap = response is more grounded in retrieved information.
    Lower overlap = response may be hallucinating.
    
    Args:
        response: Generated response
        contexts: Retrieved context chunks
    
    Returns:
        Overlap score 0.0-1.0
    """
    # Tokenize and normalize
    response_words = set(_normalize_text(response))
    context_words = set(_normalize_text(" ".join(contexts)))
    
    if not response_words or not context_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = response_words & context_words
    union = response_words | context_words
    
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Also calculate what % of response words are in context
    response_coverage = len(intersection) / len(response_words) if response_words else 0.0
    
    # Weighted average (favor coverage slightly)
    overlap_score = (0.4 * jaccard) + (0.6 * response_coverage)
    
    return min(overlap_score, 1.0)


def _normalize_text(text: str) -> List[str]:
    """
    Normalize text for overlap calculation.
    
    Args:
        text: Input text
    
    Returns:
        List of normalized tokens
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special chars
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Extract words
    words = re.findall(r'\b[a-z0-9]{2,}\b', text)  # Min 2 chars
    
    # Remove common stop words that don't help overlap
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    return [w for w in words if w not in stop_words]


def wrap_low_confidence_response(response: str, confidence_data: Dict, kind: str = "brochure") -> str:
    """
    Wrap low-confidence responses with caveats.
    
    Args:
        response: Generated response text
        confidence_data: Confidence metrics
        kind: Response type ("brochure", "wiki", or "default")
    """
    confidence = confidence_data.get("confidence", 1.0)
    reliability = confidence_data.get("reliability_tier", "high")
    
    # Only wrap low-confidence BROCHURE responses
    if kind == "brochure" and confidence < 0.3 and reliability == "low":
        caveat = (
            "I don't have sufficient information in the brochure to answer this question "
            "with confidence. I recommend contacting EngagePro directly at their Singapore "
            "office (International Business Park) for accurate information.\n\n"
        )
        return f"{caveat}{response}"
    
    # For wiki or other modes, don't add brochure-specific caveat
    return response



def generate_confidence_explanation(confidence_data: Dict) -> str:
    """
    Generate human-readable explanation of confidence score.
    Useful for debugging or showing in UI.
    
    Args:
        confidence_data: Output from calculate_confidence_score()
    
    Returns:
        Explanation string
    """
    parts = []
    
    parts.append(f"Confidence: {confidence_data['confidence']:.1%} ({confidence_data['reliability_tier']})")
    parts.append(f"Context support: {confidence_data['context_support']:.1%}")
    
    if confidence_data["has_uncertainty"]:
        parts.append("⚠️ Contains uncertainty phrases")
    
    if confidence_data["has_speculation"]:
        parts.append("⚠️ Contains speculative language")
    
    if confidence_data["has_hallucination_flags"]:
        parts.append("🚨 Potential hallucination detected")
    
    return " | ".join(parts)
