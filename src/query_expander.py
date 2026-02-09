"""
query_expander.py
Expand short/ambiguous queries using LLM before retrieval

✅ ENHANCEMENT 2: Improves retrieval for short queries
- "platforms?" → "What platforms does InnovaBot integrate with like Slack and Teams"
- "benefits?" → "What are the benefits of CX Transformer for customer service"
- "Fortune 500?" → "Which EngagePro product is used by Fortune 500 companies"
"""

from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv  # ✅ ADD THIS

load_dotenv()  # ✅ ADD THIS - Load .env file

_expander_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def should_expand_query(query: str) -> bool:
    """
    Determine if query is short/ambiguous enough to benefit from expansion.
    """
    clean_query = query.replace('?', '').replace('!', '').replace('.', '').strip()
    word_count = len(clean_query.split())
    
    # Expand if query is very short (≤ 4 words)  # ← Changed from 5
    if word_count <= 4:
        return True
    
    # Expand if single question word + 1-2 words (≤ 3 total)  # ← Changed threshold
    question_words = ['what', 'which', 'how', 'when', 'where', 'who', 'why']
    has_question = any(qw in query.lower() for qw in question_words)
    if has_question and word_count <= 3:  # ← Changed from 7
        return True
    
    return False



def expand_query(
    query: str, 
    history: List[Dict[str, str]] = None,
    max_expansions: int = 2
) -> List[str]:
    """
    Generate expanded query variations to improve retrieval.
    
    For short queries like "platforms?", generates variations like:
    - "What platforms does InnovaBot integrate with?"
    - "InnovaBot platform compatibility Slack Microsoft Teams SharePoint"
    
    Args:
        query: Original user query
        history: Conversation history for context
        max_expansions: Max number of expansions to generate (default 2)
    
    Returns:
        List of queries: [original, expansion1, expansion2, ...]
    """
    # Don't expand if query is already detailed
    if not should_expand_query(query):
        return [query]
    
    if history is None:
        history = []
    
    system_prompt = """You are a query expansion assistant for EngagePro's chatbot.

Given a short/ambiguous user query, generate 1-2 alternative phrasings that:
1. Preserve the original intent
2. Add relevant EngagePro keywords for better retrieval
3. Make the query more specific and detailed

EngagePro Products:
- InnovaBot: AI knowledge management chatbot, integrates with Slack/Teams/SharePoint/Confluence, Fortune 500 companies, 24/7 support, NLP
- CX Transformer: Customer service automation, reduces resolution times 40%, NLP, sentiment analysis, omnichannel, real-time analytics
- AI Engagement Lab: R&D initiative, productivity automation, workflow automation, cutting-edge AI research

Examples:

Query: "platforms?"
Expansions:
1. What platforms and integrations does InnovaBot support like Slack or Microsoft Teams
2. InnovaBot platform compatibility SharePoint Confluence integrations

Query: "benefits of CX?"
Expansions:
1. What are the key benefits and features of CX Transformer for customer service automation
2. CX Transformer advantages NLP sentiment analysis 40% resolution time improvements

Query: "Fortune 500?"
Expansions:
1. Which EngagePro product is trusted by Fortune 500 companies
2. InnovaBot Fortune 500 enterprise customers adoption

Query: "how many products?"
Expansions:
1. How many products and services does EngagePro offer in total
2. Complete EngagePro product lineup InnovaBot CX Transformer AI Engagement Lab

Query: "40%"
Expansions:
1. What does the 40% improvement metric refer to in EngagePro products
2. CX Transformer 40% reduction in customer service resolution times

Return ONLY a numbered list (1-2 items), no extra text or explanations."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent conversation history for context-aware expansion
    for h in history[-4:]:  # Last 2 exchanges (4 messages)
        messages.append({
            "role": h["role"], 
            "content": h["content"][:200]  # Truncate long messages
        })
    
    messages.append({
        "role": "user", 
        "content": f"Expand this query:\n{query}"
    })
    
    try:
        resp = _expander_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,  # More creative for variations
            max_tokens=100,
        )
        
        expansions_text = resp.choices[0].message.content.strip()
        
        # Parse numbered list
        import re
        expansions = re.findall(r'\d+\.\s*(.+)', expansions_text)
        
        # Filter out empty expansions
        expansions = [exp.strip() for exp in expansions if exp.strip()]
        
        # Return original + expansions (limited to max_expansions)
        all_queries = [query] + expansions[:max_expansions]
        
        print(f"🔄 Query expansion: {len(all_queries)} variations")
        for i, q in enumerate(all_queries):
            marker = "📌" if i == 0 else "  "
            print(f"   {marker} {i+1}. {q}")
        
        return all_queries
    
    except Exception as e:
        print(f"⚠️  Query expansion failed: {e}")
        return [query]  # Fall back to original
