"""
validators.py
Post-generation validation to ensure response quality and completeness
"""

from typing import Dict, List
import re

# All EngagePro products that should be mentioned for comprehensive queries
REQUIRED_PRODUCTS = ["InnovaBot", "CX Transformer", "AI Engagement Lab"]

def validate_product_completeness(query: str, response: str, kind: str) -> Dict[str, any]:
    """
    Check if response mentions all products when query asks for complete list.
    
    This ensures that when users ask "What services does EngagePro offer?" or
    "List all products", the chatbot always mentions all three products.
    
    Args:
        query: Original user query
        response: Generated LLM response
        kind: Response type ("brochure", "wiki", or "default")
    
    Returns:
        Dict with:
            - is_complete: bool indicating if all required products mentioned
            - missing_products: list of products not mentioned
            - needs_enrichment: bool indicating if response should be enriched
    """
    # Only validate brochure responses
    if kind != "brochure":
        return {
            "is_complete": True, 
            "missing_products": [], 
            "needs_enrichment": False
        }
    
    # Patterns that indicate user wants ALL products/services listed
    completeness_triggers = [
        r"what (services|products|solutions)",
        r"list.*products",
        r"list.*services",
        r"all.*products",
        r"all.*services",
        r"all.*solutions",
        r"complete.*lineup",
        r"complete.*product",
        r"how many.*products",
        r"how many.*services",
        r"describe.*offerings",
        r"describe.*services",
        r"what can engagepro (provide|solve|offer|do)",
        r"what does engagepro (provide|offer)",
        r"tell me about engagepro'?s? products",
        r"tell me about engagepro'?s? services",
    ]
    
    query_lower = query.lower()
    requires_all_products = any(
        re.search(pattern, query_lower) for pattern in completeness_triggers
    )
    
    # If query doesn't require completeness, skip validation
    if not requires_all_products:
        return {
            "is_complete": True, 
            "missing_products": [], 
            "needs_enrichment": False
        }
    
    # Check which products are actually mentioned in the response
    missing = []
    for product in REQUIRED_PRODUCTS:
        # Case-insensitive search for product name
        if product.lower() not in response.lower():
            missing.append(product)
    
    return {
        "is_complete": len(missing) == 0,
        "missing_products": missing,
        "needs_enrichment": len(missing) > 0
    }


def enrich_response_with_missing_products(response: str, missing_products: List[str]) -> str:
    """
    Append missing product information to an incomplete response.
    
    This acts as a safety net - if the LLM forgot to mention a product,
    we add it programmatically to ensure completeness.
    
    Args:
        response: Original LLM response
        missing_products: List of products not mentioned
    
    Returns:
        Enriched response with missing products added
    """
    if not missing_products:
        return response
    
    # Detailed descriptions for each product
    product_info = {
        "InnovaBot": (
            "**InnovaBot** - An AI-powered knowledge management chatbot trusted by "
            "Fortune 500 companies. It integrates with Slack, Microsoft Teams, SharePoint, "
            "and Confluence, breaking down information silos and providing 24/7 multi-channel support."
        ),
        "CX Transformer": (
            "**CX Transformer** - A customer service automation platform that reduces "
            "resolution times by 40%. It uses NLP and sentiment analysis to improve customer "
            "satisfaction scores, offering omnichannel support with real-time analytics."
        ),
        "AI Engagement Lab": (
            "**AI Engagement Lab** - EngagePro's current R&D initiative focused on productivity "
            "and customer service automation. It reduces workloads through automated workflows "
            "and conducts cutting-edge research in AI-driven engagement."
        )
    }
    
    # Build enrichment text
    if len(missing_products) == 1:
        enrichment = f"\n\nAdditionally, EngagePro also offers {product_info[missing_products[0]]}"
    else:
        enrichment = "\n\nAdditionally, EngagePro also offers:\n\n"
        for product in missing_products:
            enrichment += f"- {product_info[product]}\n"
    
    return response + enrichment


# Optional: Validate response doesn't contain hallucinated information
def check_for_hallucinations(response: str, contexts: List[str]) -> Dict[str, any]:
    """
    Basic hallucination detection by checking if response mentions things
    not present in the retrieved contexts.
    
    This is a simple version - checks for specific known-false statements.
    
    Args:
        response: Generated response
        contexts: Retrieved context chunks
    
    Returns:
        Dict with hallucination detection results
    """
    response_lower = response.lower()
    
    # Known false information that chatbot should NOT claim
    forbidden_claims = [
        (r"engagepro is publicly traded", "EngagePro is not mentioned as publicly traded in brochure"),
        (r"engagepro was founded in \d{4}", "Founding year not in brochure (unless actually stated)"),
        (r"revenue of \$?\d+[mb]illion", "Specific revenue not in brochure (unless actually stated)"),
        (r"ceo is", "CEO name not in brochure"),
        (r"headquarters in .* except singapore", "EngagePro is in Singapore"),
    ]
    
    hallucinations_found = []
    for pattern, reason in forbidden_claims:
        if re.search(pattern, response_lower):
            # Check if this info is actually in contexts
            context_text = " ".join(contexts).lower()
            if pattern not in context_text:
                hallucinations_found.append(reason)
    
    return {
        "has_hallucinations": len(hallucinations_found) > 0,
        "hallucinations": hallucinations_found
    }