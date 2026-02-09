# guardrails.py

import re
import os
from typing import Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client for moderation
_moderation_client = None

def _get_moderation_client():
    """Lazy initialization of OpenAI client for moderation."""
    global _moderation_client
    if _moderation_client is None:
        _moderation_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _moderation_client


EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-\s]?)?(?:\d{3,4}[-\s]?){2,3}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
NRIC_PATTERN = re.compile(r"[STFG]\d{7}[A-Z]", re.IGNORECASE)



def _mask_email(text: str) -> str:
    def repl(m: re.Match) -> str:
        value = m.group(0)
        user, _, domain = value.partition("@")
        if not domain:
            return "[EMAIL]"
        if len(user) <= 2:
            masked_user = "*" * len(user)
        else:
            masked_user = user[0] + "*" * (len(user) - 2) + user[-1]
        return masked_user + "@***"

    return EMAIL_PATTERN.sub(repl, text)



def _mask_phone(text: str) -> str:
    def repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) <= 4:
            return "[PHONE]"
        masked = "*" * (len(digits) - 4) + digits[-4:]
        return masked

    return PHONE_PATTERN.sub(repl, text)



def _mask_credit_card(text: str) -> str:
    def repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) < 13:
            return "[CARD]"
        masked = "*" * (len(digits) - 4) + digits[-4:]
        return masked

    return CREDIT_CARD_PATTERN.sub(repl, text)



def _mask_nric(text: str) -> str:
    return NRIC_PATTERN.sub("[NRIC]", text)



def mask_pii(text: str) -> str:
    """
    Shared masking logic used for both input and output (if needed).
    """
    text = _mask_email(text)
    text = _mask_phone(text)
    text = _mask_credit_card(text)
    text = _mask_nric(text)
    return text



def contains_pii(text: str) -> bool:
    """
    Lightweight check to see if text likely contains PII.
    """
    return bool(
        EMAIL_PATTERN.search(text)
        or PHONE_PATTERN.search(text)
        or CREDIT_CARD_PATTERN.search(text)
        or NRIC_PATTERN.search(text)
    )



# -------- INPUT GUARDRAIL -------------------------------------------------------



def sanitize_input(text: str) -> tuple:
    """
    Sanitize user input by masking PII.
    
    Args:
        text: User input text
        
    Returns:
        Tuple of (sanitized_text, had_pii)
    """
    had_pii = contains_pii(text)
    
    # Mask NRIC/FIN
    text = NRIC_PATTERN.sub('[NRIC]', text)
    
    # Mask credit cards
    text = CREDIT_CARD_PATTERN.sub('[CREDIT_CARD]', text)
    
    # Mask emails
    text = EMAIL_PATTERN.sub('[EMAIL]', text)
    
    # Mask phone numbers
    text = PHONE_PATTERN.sub('[PHONE]', text)
    
    return text, had_pii



# -------- CONTENT SAFETY GUARDRAIL ----------------------------------------------



def is_content_unsafe(text: str) -> Tuple[bool, str]:
    """
    Use OpenAI Moderation API to detect unsafe/inappropriate content.
    
    Detects:
    - Sexual content
    - Hate speech
    - Harassment
    - Self-harm
    - Violence
    - Other harmful content
    
    Args:
        text: User query text
        
    Returns:
        Tuple of (is_unsafe, reason)
        - is_unsafe: True if content is flagged as unsafe
        - reason: Comma-separated list of flagged categories
    """
    try:
        client = _get_moderation_client()
        response = client.moderations.create(input=text)
        result = response.results[0]
        
        if result.flagged:
            # Get which categories were flagged
            categories_dict = result.categories.model_dump()
            flagged_categories = [
                category for category, is_flagged in categories_dict.items() 
                if is_flagged
            ]
            reason = ", ".join(flagged_categories)
            print(f"DEBUG: Content flagged for: {reason}")
            return True, reason
        
        return False, ""
        
    except Exception as e:
        print(f"DEBUG: Moderation API error: {e}")
        # Fail open - don't block if API fails
        return False, ""



# -------- OUTPUT GUARDRAIL ------------------------------------------------------



def sanitize_output(model_text: str) -> str:
    """
    Output guardrail (disabled for now): return text unchanged so brochure
    content (emails, phones, metrics, URLs) is not altered. PII protection
    is enforced at input time and via routing.
    """
    return model_text
