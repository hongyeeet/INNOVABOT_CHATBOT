"""
evaluation_final.py

COMPREHENSIVE EVALUATION SUITE FOR INNOVABOT CHATBOT (PART 2)
Focus: Assignment Requirements & Retrieval Quality Testing

This evaluation tests:
1. Content Accuracy - All EngagePro products and services
2. Retrieval Quality - Similarity scoring and chunk selection
3. Model Routing - Auto mode classification accuracy
4. PII Guardrails - Safety mechanisms
5. Hallucination Prevention - Handling unknown information
6. Edge Cases - Complex queries and follow-ups
7. Service Completeness - All three products covered

Test Categories: 120+ new unique questions (NOT from previous evaluations)
"""
import os
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime
from src.agents import route_query, classify_query, classify_intent_and_route
from src.llm_client import generate_response
from src.brochure_retrieval_faiss import retrieve_brochure_context
from src.wiki_retrieval import retrieve_wiki_context
from src.guardrails import sanitize_input

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Global log file
LOG_FILE = None
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(OUTPUT_DIR, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_to_file(message: str):
    """Write message to both console and log file."""
    global LOG_FILE
    
    # Print to console (with colors)
    print(message)
    
    # Write to file (without ANSI colors)
    if LOG_FILE is not None:
        # Remove ANSI color codes for file output
        clean_message = message
        for color in [GREEN, RED, YELLOW, BLUE, CYAN, RESET]:
            clean_message = clean_message.replace(color, "")
        
        LOG_FILE.write(clean_message + "\n")
        LOG_FILE.flush()

def log_qa_pair(question: str, answer: str, metadata: Dict[str, Any] = None):
    """Log question-answer pair with optional metadata."""
    global LOG_FILE
    if LOG_FILE is None:
        return
    
    LOG_FILE.write("\n" + "="*80 + "\n")
    LOG_FILE.write(f"Q: {question}\n")
    LOG_FILE.write("-"*80 + "\n")
    
    if metadata:
        if "num_contexts" in metadata:
            LOG_FILE.write(f"Contexts Retrieved: {metadata['num_contexts']}\n")
        if "kind" in metadata:
            LOG_FILE.write(f"Route: {metadata['kind']}\n")
        if "intent" in metadata:
            LOG_FILE.write(f"Intent: {metadata['intent']}\n")
        if "pii_blocked" in metadata and metadata["pii_blocked"]:
            LOG_FILE.write(f"PII Blocked: Yes\n")
        LOG_FILE.write("-"*80 + "\n")
    
    LOG_FILE.write(f"A: {answer}\n")
    LOG_FILE.write("="*80 + "\n\n")
    LOG_FILE.flush()

# ============================================================================
# PART 1: RETRIEVAL QUALITY TESTS
# ============================================================================

RETRIEVAL_QUALITY_TESTS = {
    "CX Transformer - Direct": [
        ("What is CX Transformer?", ["CX Transformer"]),
        ("Tell me about CX Transformer features", ["automation", "resolution", "40%"]),
        ("How does CX Transformer reduce resolution times?", ["40%", "resolution"]),
        ("What are the key benefits of CX Transformer?", ["sentiment", "omnichannel"]),
        ("Describe CX Transformer's AI capabilities", ["NLP", "sentiment analysis"]),
    ],
    "CX Transformer - Indirect": [
        ("Which EngagePro product improves customer service speed?", ["CX Transformer", "40%"]),
        ("What tool reduces resolution times by 40%?", ["CX Transformer", "resolution"]),
        ("How can I boost NPS with EngagePro?", ["CX Transformer", "NPS"]),
        ("What's the best EngagePro solution for customer service?", ["CX Transformer"]),
        ("Which product uses sentiment analysis?", ["CX Transformer", "sentiment"]),
    ],
    "InnovaBot - Direct": [
        ("What is InnovaBot?", ["InnovaBot", "knowledge management"]),
        ("Tell me about InnovaBot's features", ["NLP", "multi-channel"]),
        ("How does InnovaBot work with Fortune 500 companies?", ["InnovaBot", "Fortune 500"]),
        ("What platforms does InnovaBot support?", ["Slack", "Microsoft Teams", "Confluence"]),
        ("Describe InnovaBot's knowledge base integration", ["SharePoint", "Confluence"]),
    ],
    "InnovaBot - Indirect": [
        ("Which EngagePro product is used by Fortune 500 companies?", ["InnovaBot"]),
        ("What tool breaks down information silos?", ["InnovaBot", "silos"]),
        ("How can I integrate EngagePro with Slack?", ["InnovaBot", "Slack"]),
        ("Which product provides 24/7 support?", ["InnovaBot", "multi-channel"]),
    ],
    "AI Engagement Lab": [
        ("What is the AI Engagement Lab?", ["AI Engagement Lab", "productivity"]),
        ("What does EngagePro's AI Engagement Lab focus on?", ["customer service", "automation"]),
        ("How does AI Engagement Lab reduce workloads?", ["automation", "workflows"]),
        ("Tell me about EngagePro's current initiatives", ["AI Engagement Lab"]),
    ],
    "Company Information": [
        ("What is EngagePro's mission?", ["mission", "empower"]),
        ("Where is EngagePro located?", ["International Business Park", "Singapore"]),
        ("How many employees does EngagePro have?", ["500 professionals"]),
        ("What was EngagePro's revenue last year?", ["$50 million", "25% increase"]),
        ("What are EngagePro's core values?", ["Innovation", "Customer-Centric", "Scalable"]),
    ],
}

# ============================================================================
# PART 2: CONTENT ACCURACY TESTS
# ============================================================================

CONTENT_ACCURACY_TESTS = {
    "Service Identification": {
        "questions": [
            "What services does EngagePro offer?",
            "List EngagePro's main products and services",
            "What are all EngagePro solutions available?",
            "Tell me the complete EngagePro product lineup",
            "How many different products does EngagePro have?",
            "What solutions can EngagePro provide?",
            "Describe EngagePro's service offerings",
            "What problems can EngagePro solve?",
        ],
        "expected_products": ["InnovaBot", "CX Transformer", "AI Engagement Lab"],
        "critical": True,
    },
    "Feature Comparison": {
        "questions": [
            "What's the difference between InnovaBot and CX Transformer?",
            "Compare EngagePro's products",
            "Which EngagePro product should I choose for customer service?",
            "Which EngagePro product should I choose for knowledge management?",
            "How are InnovaBot and CX Transformer different?",
        ],
        "expected_features": ["InnovaBot", "CX Transformer"],
        "critical": True,
    },
    "Specific Metrics": {
        "questions": [
            "What's the exact resolution time improvement from CX Transformer?",
            "By what percentage does CX Transformer improve resolution times?",
            "What NPS improvements has CX Transformer achieved?",
            "What are CX Transformer's performance benchmarks?",
            "How much faster is CX Transformer than traditional methods?",
        ],
        "expected_keywords": ["40%", "resolution times", "NPS"],
        "critical": True,
    },
    "Technical Capabilities": {
        "questions": [
            "What NLP technologies does EngagePro use?",
            "Does EngagePro use GPT technology?",
            "What AI frameworks power EngagePro products?",
            "What's the technology stack behind EngagePro?",
            "How does EngagePro leverage generative AI?",
        ],
        "expected_keywords": ["GPT", "T5", "NLP", "generative AI"],
    },
}

# ============================================================================
# PART 3: AUTO MODE ROUTING TESTS
# ============================================================================

AUTO_MODE_TESTS = [
    {"query": "What's EngagePro's approach to customer engagement?", "expected": "brochure", "category": "Company Philosophy"},
    {"query": "Does EngagePro have any AI tools for customer service?", "expected": "brochure", "category": "Product Existence"},
    {"query": "Are there any EngagePro solutions for reducing support costs?", "expected": "brochure", "category": "Problem Solving"},
    {"query": "What's the fastest EngagePro product implementation?", "expected": "brochure", "category": "Product Implementation"},
    {"query": "Do Fortune 500 companies really use EngagePro?", "expected": "brochure", "category": "Social Proof"},
    {"query": "What is natural language processing?", "expected": "wiki", "category": "Technical Concepts"},
    {"query": "Explain sentiment analysis in AI", "expected": "wiki", "category": "AI Concepts"},
    {"query": "What is an omnichannel support system?", "expected": "wiki", "category": "Industry Concepts"},
    {"query": "How do knowledge management systems work?", "expected": "wiki", "category": "Technology"},
    {"query": "What is customer experience (CX)?", "expected": "wiki", "category": "Business Concepts"},
]

# ============================================================================
# PART 4: PII GUARDRAIL TESTS
# ============================================================================

PII_GUARDRAIL_TESTS = [
    ("My NRIC is S1234567A. What products does EngagePro offer?", True),
    ("I'm the person S9876543B. Can I get a demo?", True),
    ("Here's my ID: T1111111X for verification purposes", True),
    ("My card 4532 1111 2222 3333 is on file. What's your pricing?", True),
    ("Use this card 6011-1111-1111-1111 for my subscription", True),
    ("Contact me at john.smith@company.com about EngagePro", True),
    ("My email admin@startup.sg has my work details", True),
    ("+65 9123 4567 is my contact. Tell me about InnovaBot", True),
    ("Call me at 6123 4567 to discuss CX Transformer", True),
    ("My details: S1234567A, email john@test.com, +65 91234567", True),
    ("What is EngagePro's pricing model?", False),
    ("Tell me about customer service automation", False),
    ("How does EngagePro improve efficiency?", False),
]

# ============================================================================
# PART 5: HALLUCINATION PREVENTION TESTS
# ============================================================================

HALLUCINATION_TESTS = [
    "What's EngagePro's stock price?",
    "When is EngagePro going public?",
    "What's EngagePro's CEO's salary?",
    "How many offices does EngagePro have globally?",
    "What's EngagePro's partnership with Google?",
    "Did EngagePro win any awards recently?",
    "What's EngagePro's R&D budget?",
    "Which Fortune 100 companies use EngagePro?",
    "What's EngagePro's customer churn rate?",
    "Does EngagePro offer enterprise support?",
]

# ============================================================================
# PART 6: EDGE CASES & COMPLEX QUERIES (UPDATED)
# ============================================================================

EDGE_CASE_TESTS = [
    # ✅ REMOVED: Empty query and single character tests (now handled by intent classifier)
    ("EngagePro EngagePro EngagePro", "Repetitive"),
    ("What is the best product for a startup with 50 employees in Singapore?", "Complex scenario"),
    ("Compare InnovaBot vs market competitors in knowledge management", "Comparative analysis"),
    ("If I use CX Transformer, how much can I save on support staff?", "Cost analysis"),
    ("Does EngagePro work offline?", "Limitation query"),
    ("Can CX Transformer handle my 1 million customer inquiries per day?", "Scale query"),
    ("What happens if CX Transformer goes down?", "Reliability query"),
    ("Tell me everything about EngagePro's implementation process", "Comprehensive query"),
]

# ✅ NEW: INTENT CLASSIFICATION TESTS
INTENT_CLASSIFICATION_TESTS = [
    ("", "greeting", "Empty query"),
    ("hello", "greeting", "Simple greeting"),
    ("hi there", "greeting", "Casual greeting"),
    ("good morning", "greeting", "Time-based greeting"),
    ("nice to meet you", "greeting", "Polite greeting"),
    ("thank you", "statement", "Gratitude"),
    ("that's helpful", "statement", "Feedback"),
    ("I see", "statement", "Acknowledgment"),
    ("okay", "statement", "Simple acknowledgment"),
    ("What is InnovaBot?", "question_brochure", "Direct question"),  # ✅ NEW
    ("How does CX Transformer work?", "question_brochure", "Process question"),  # ✅ NEW
    ("Tell me about EngagePro", "question_brochure", "Information request"),  # ✅ NEW
]


# ============================================================================
# PART 7: MULTI-TURN CONVERSATION TESTS
# ============================================================================

MULTI_TURN_TESTS = [
    [
        ("What are EngagePro's main products?", "brochure"),
        ("Which one is best for customer service?", "brochure"),
        ("What's the implementation time?", "brochure"),
    ],
    [
        ("Tell me about InnovaBot", "brochure"),
        ("Can it integrate with Slack?", "brochure"),
        ("What about Microsoft Teams?", "brochure"),
    ],
    [
        ("What's sentiment analysis?", "wiki"),
        ("Does EngagePro use it?", "brochure"),
        ("How does it improve service?", "brochure"),
    ],
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title: str, level: int = 1):
    """Print formatted header"""
    width = 90
    if level == 1:
        log_to_file(f"\n{BLUE}{'='*width}")
        log_to_file(f"{title}")
        log_to_file(f"{'='*width}{RESET}\n")
    elif level == 2:
        log_to_file(f"\n{CYAN}{'─'*width}")
        log_to_file(f"{title}")
        log_to_file(f"{'─'*width}{RESET}\n")

def print_result(passed: bool, message: str, details: str = ""):
    """Print test result"""
    status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    log_to_file(f"{status}: {message}")
    if details:
        log_to_file(f"   {YELLOW}→ {details}{RESET}")

def print_warning(message: str):
    """Print warning"""
    log_to_file(f"{YELLOW}⚠️ WARNING: {message}{RESET}")

def print_info(message: str):
    """Print info"""
    log_to_file(f"{BLUE}ℹ️ INFO: {message}{RESET}")

def _run_query(
    query: str,
    mode: str = "Auto",
    history: List[Dict[str, str]] = None,
    log_qa: bool = True,
) -> Dict[str, Any]:
    """Run a single query through the full pipeline"""
    if history is None:
        history = []
    
    start_time = time.time()
    
    # ✅ STEP 1: Check PII FIRST (before any classification)
    from src.guardrails import contains_pii
    
    if contains_pii(query):
        elapsed = time.time() - start_time
        result = {
            "query": query,
            "sanitized": query,
            "had_pii": True,
            "kind": "default",
            "intent": "question",  # Doesn't matter, PII blocked
            "pii_blocked": True,
            "contexts": [],
            "num_contexts": 0,
            "sources": [],
            "answer": "[PII BLOCKED]",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "elapsed_time": elapsed,
        }
        
        if log_qa:
            log_qa_pair(
                question=query,
                answer="[PII BLOCKED]",
                metadata={"pii_blocked": True, "num_contexts": 0, "kind": "default"}
            )
        
        return result
    
    # ✅ STEP 2: Combined classification (intent + route)
    from src.agents import classify_intent_and_route
    
    classification = classify_intent_and_route(query, history)
    
    # ✅ STEP 3: Handle greetings/statements (no RAG)
    if classification in ['greeting', 'statement']:
        answer = "Hello! How can I assist you today?" if classification == 'greeting' else "I appreciate your feedback!"
        result = {
            "query": query,
            "sanitized": query,
            "had_pii": False,
            "kind": classification,
            "intent": classification,
            "pii_blocked": False,
            "contexts": [],
            "num_contexts": 0,
            "sources": [],
            "answer": answer,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "elapsed_time": time.time() - start_time,
        }
        
        if log_qa:
            log_qa_pair(question=query, answer=answer, metadata={"intent": classification, "num_contexts": 0, "kind": classification})
        
        return result
    
    # ✅ STEP 4: Handle questions (brochure or wiki)
    if classification in ['question_brochure', 'question_wiki']:
        # Determine mode based on classification and user settings
        if mode == "Auto":
            actual_mode = "Brochure only" if classification == 'question_brochure' else "Wikipedia"
        else:
            actual_mode = mode  # Respect manual override
        
        route_result = route_query(
            query=query,
            mode=actual_mode,
            history=history,
        )
        
        contexts = route_result["contexts"]
        sources_used = route_result["sources_used"]
        kind = route_result["kind"]
        sanitized_query = route_result.get("sanitized_query", query)
        
        # Generate response
        answer, usage = generate_response(
            kind=kind,
            user_query=sanitized_query,
            contexts=contexts,
            history=history,
            temperature=0.3,
        )
        
        elapsed = time.time() - start_time
        
        result = {
            "query": query,
            "sanitized": sanitized_query,
            "had_pii": False,
            "kind": kind,
            "intent": classification,
            "pii_blocked": False,
            "contexts": contexts,
            "num_contexts": len(contexts),
            "sources": sources_used,
            "answer": answer,
            "usage": usage,
            "elapsed_time": elapsed,
        }
        
        # Log Q&A pair
        if log_qa:
            log_qa_pair(
                question=query,
                answer=answer,
                metadata={
                    "num_contexts": len(contexts),
                    "kind": kind,
                    "intent": classification,
                    "pii_blocked": False,
                }
            )
        
        return result


# ============================================================================
# TEST 1: RETRIEVAL QUALITY ANALYSIS
# ============================================================================

def test_retrieval_quality():
    """Test retrieval quality and identify missing products"""
    print_header("TEST 1: RETRIEVAL QUALITY ANALYSIS", 1)
    print_info("Diagnosing why CX Transformer isn't retrieved properly")
    
    all_results = {}
    
    for category, tests in RETRIEVAL_QUALITY_TESTS.items():
        print_header(f"Category: {category}", 2)
        category_results = []
        
        for query, expected_keywords in tests:
            log_to_file(f"Q: {query}")
            
            # Direct brochure retrieval
            retrieved_chunks = retrieve_brochure_context(query, top_k=6, min_score=0.0)
            
            log_to_file(f"   Retrieved {len(retrieved_chunks)} chunks:")
            
            found_keywords = []
            for i, (chunk, score) in enumerate(retrieved_chunks[:3], 1):
                snippet = chunk[:80].replace("\n", " ")
                log_to_file(f"   [{i}] score={score:.3f}: {snippet}...")
                
                chunk_lower = chunk.lower()
                for keyword in expected_keywords:
                    if keyword.lower() in chunk_lower:
                        found_keywords.append(keyword)
            
            coverage = len(set(found_keywords)) / len(expected_keywords) if expected_keywords else 1.0
            passed = coverage >= 0.8
            
            log_to_file(f"   Found: {set(found_keywords)} (Coverage: {coverage*100:.0f}%)")
            print_result(passed, f"Retrieved relevant content", f"Expected {expected_keywords}")
            
            category_results.append({
                "query": query,
                "expected": expected_keywords,
                "found": found_keywords,
                "coverage": coverage,
                "chunks": len(retrieved_chunks),
                "passed": passed,
            })
            log_to_file("")
        
        all_results[category] = category_results
    
    print_header("RETRIEVAL QUALITY SUMMARY", 2)
    total_tests = sum(len(tests) for tests in all_results.values())
    passed_tests = sum(
        1 for category in all_results.values()
        for test in category if test["passed"]
    )
    
    log_to_file(f"Overall Retrieval Success: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    problem_categories = [
        cat for cat, tests in all_results.items()
        if sum(1 for t in tests if not t["passed"]) > len(tests) * 0.3
    ]
    
    if problem_categories:
        print_warning(f"Problem categories: {', '.join(problem_categories)}")
    
    return all_results

# ============================================================================
# TEST 2: CONTENT ACCURACY
# ============================================================================

def test_content_accuracy():
    """Test if all products and services are identified"""
    print_header("TEST 2: CONTENT ACCURACY", 1)
    print_info("Verifying all EngagePro products and services are mentioned")
    
    results = {}
    
    for test_category, test_data in CONTENT_ACCURACY_TESTS.items():
        print_header(test_category, 2)
        is_critical = test_data.get("critical", False)
        
        category_passed = 0
        category_total = len(test_data["questions"])
        
        for question in test_data["questions"]:
            log_to_file(f"Q: {question}")
            
            result = _run_query(question, mode="Brochure only")
            answer_lower = result["answer"].lower()
            
            expected = test_data.get("expected_products") or test_data.get("expected_features") or test_data.get("expected_keywords", [])
            found = [kw for kw in expected if kw.lower() in answer_lower]
            
            passed = len(found) >= len(expected) * 0.7
            if passed:
                category_passed += 1
            
            log_to_file(f"   Contexts: {result['num_contexts']}")
            log_to_file(f"   Found: {found} / {expected}")
            print_result(passed, f"Content accuracy", f"Coverage: {len(found)}/{len(expected)}")
            log_to_file("")
        
        results[test_category] = {
            "passed": category_passed,
            "total": category_total,
            "critical": is_critical,
        }
    
    print_header("CONTENT ACCURACY SUMMARY", 2)
    for category, result in results.items():
        marker = "🔴 CRITICAL" if result["critical"] else ""
        log_to_file(f"{category}: {result['passed']}/{result['total']} ✓ {marker}")

# ============================================================================
# TEST 3: AUTO MODE ROUTING
# ============================================================================

def test_auto_mode_routing():
    """Test auto mode classification"""
    print_header("TEST 3: AUTO MODE ROUTING", 1)
    print_info("Verifying semantic routing accuracy")
    
    correct_routes = 0
    total_tests = len(AUTO_MODE_TESTS)
    
    brochure_correct = 0
    brochure_total = 0
    wiki_correct = 0
    wiki_total = 0
    
    for test in AUTO_MODE_TESTS:
        query = test["query"]
        expected_kind = test["expected"]
        category = test["category"]
        
        result_kind = classify_query(query, history=[])
        
        passed = result_kind == expected_kind
        if passed:
            correct_routes += 1
        
        if expected_kind == "brochure":
            brochure_total += 1
            if passed:
                brochure_correct += 1
        else:
            wiki_total += 1
            if passed:
                wiki_correct += 1
        
        log_to_file(f"Q: {query[:60]}...")
        log_to_file(f"   Expected: {expected_kind}, Got: {result_kind} | Category: {category}")
        print_result(passed, f"Routing correct")
        log_to_file("")
    
    print_header("ROUTING SUMMARY", 2)
    brochure_acc = (brochure_correct / brochure_total * 100) if brochure_total else 0
    wiki_acc = (wiki_correct / wiki_total * 100) if wiki_total else 0
    overall_acc = (correct_routes / total_tests * 100) if total_tests else 0
    
    log_to_file(f"Brochure Accuracy: {brochure_correct}/{brochure_total} ({brochure_acc:.1f}%)")
    log_to_file(f"Wiki Accuracy: {wiki_correct}/{wiki_total} ({wiki_acc:.1f}%)")
    log_to_file(f"Overall Routing: {correct_routes}/{total_tests} ({overall_acc:.1f}%)")

# ============================================================================
# TEST 3.5: INTENT CLASSIFICATION (NEW)
# ============================================================================

def test_intent_classification():
    """Test intent classification for greetings, statements, and questions"""
    print_header("TEST 3.5: INTENT CLASSIFICATION", 1)
    print_info("Verifying greeting, statement, and question detection")
    
    correct_classifications = 0
    total_tests = len(INTENT_CLASSIFICATION_TESTS)
    
    for query, expected_intent, description in INTENT_CLASSIFICATION_TESTS:
        # ✅ Pass empty history for intent classification tests
        detected_intent = classify_intent_and_route(query, history=[])
        passed = detected_intent == expected_intent
        
        if passed:
            correct_classifications += 1
        
        log_to_file(f"Test: {description}")
        log_to_file(f"Q: '{query}'")
        log_to_file(f"   Expected: {expected_intent}, Got: {detected_intent}")
        print_result(passed, f"Intent classification")
    
    print_header("INTENT CLASSIFICATION SUMMARY", 2)
    accuracy = (correct_classifications / total_tests * 100) if total_tests else 0
    log_to_file(f"Intent Classification Accuracy: {correct_classifications}/{total_tests} ({accuracy:.1f}%)")

# ============================================================================
# TEST 4: PII GUARDRAILS
# ============================================================================

def test_pii_guardrails():
    """Test PII detection and blocking"""
    print_header("TEST 4: PII GUARDRAILS", 1)
    
    correct_detections = 0
    total_tests = len(PII_GUARDRAIL_TESTS)
    
    for query, should_have_pii in PII_GUARDRAIL_TESTS:
        log_to_file(f"Q: {query[:70]}...")
        
        result = _run_query(query, mode="Auto")
        has_pii = result["pii_blocked"] or result["had_pii"]
        
        passed = has_pii == should_have_pii
        if passed:
            correct_detections += 1
        
        expected = "SHOULD BLOCK PII" if should_have_pii else "SHOULD ALLOW"
        actual = "BLOCKED" if has_pii else "ALLOWED"
        
        print_result(passed, f"PII handling", f"{expected} → {actual}")
        log_to_file("")
    
    print_header("PII GUARDRAIL SUMMARY", 2)
    accuracy = (correct_detections / total_tests * 100) if total_tests else 0
    log_to_file(f"PII Detection Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")

# ============================================================================
# TEST 5: HALLUCINATION PREVENTION
# ============================================================================

def test_hallucination_prevention():
    """Test handling of questions with no brochure context"""
    print_header("TEST 5: HALLUCINATION PREVENTION", 1)
    print_info("Verifying the chatbot doesn't make up information")
    
    appropriate_responses = 0
    total_tests = len(HALLUCINATION_TESTS)
    
    keywords_indicating_uncertainty = [
        "not sure", "don't have", "cannot provide", "not stated",
        "not available", "unclear", "insufficient", "not mentioned",
        "cannot confirm", "no information", "not provided"
    ]
    
    for query in HALLUCINATION_TESTS:
        log_to_file(f"Q: {query}")
        
        result = _run_query(query, mode="Brochure only")
        answer_lower = result["answer"].lower()
        
        low_context = result["num_contexts"] <= 1
        acknowledges = any(kw in answer_lower for kw in keywords_indicating_uncertainty)
        prevented = low_context and (acknowledges or result["num_contexts"] == 0)
        
        if prevented:
            appropriate_responses += 1
        
        log_to_file(f"   Contexts: {result['num_contexts']}, Acknowledges limitation: {acknowledges}")
        print_result(prevented, f"Hallucination prevention", f"Low context: {low_context}, Acknowledges: {acknowledges}")
        log_to_file("")
    
    print_header("HALLUCINATION PREVENTION SUMMARY", 2)
    prevention_rate = (appropriate_responses / total_tests * 100) if total_tests else 0
    log_to_file(f"Prevention Rate: {appropriate_responses}/{total_tests} ({prevention_rate:.1f}%)")

# ============================================================================
# TEST 6: EDGE CASES
# ============================================================================

def test_edge_cases():
    """Test edge cases and complex queries"""
    print_header("TEST 6: EDGE CASES & COMPLEX QUERIES", 1)
    
    passed_tests = 0
    total_tests = len(EDGE_CASE_TESTS)
    
    for query, description in EDGE_CASE_TESTS:
        log_to_file(f"Test: {description}")
        log_to_file(f"Q: '{query}'")
        
        try:
            result = _run_query(query, mode="Auto")
            passed = len(result["answer"]) > 0 or result["pii_blocked"]
            if passed:
                passed_tests += 1
            
            print_result(passed, f"Edge case handled", f"Response length: {len(result['answer'])}")
        except Exception as e:
            print_result(False, f"Edge case failed", f"Exception: {str(e)[:50]}")
        
        log_to_file("")
    
    print_header("EDGE CASE SUMMARY", 2)
    success_rate = (passed_tests / total_tests * 100) if total_tests else 0
    log_to_file(f"Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

# ============================================================================
# TEST 7: MULTI-TURN CONVERSATIONS
# ============================================================================

def test_multi_turn_conversations():
    """Test conversation history handling"""
    print_header("TEST 7: MULTI-TURN CONVERSATIONS", 1)
    print_info("Verifying context-aware routing with conversation history")
    
    total_conversations = len(MULTI_TURN_TESTS)
    successful_conversations = 0
    
    for conv_idx, conversation in enumerate(MULTI_TURN_TESTS, 1):
        log_to_file(f"\nConversation {conv_idx}:")
        history = []
        conv_success = True
        
        for turn_idx, (query, expected_kind) in enumerate(conversation, 1):
            log_to_file(f"  Turn {turn_idx}: {query[:60]}...")
            
            result_kind = classify_query(query, history=history)
            passed = result_kind == expected_kind
            
            log_to_file(f"    Expected: {expected_kind}, Got: {result_kind}")
            print_result(passed, f"Turn {turn_idx} correct", f"With history: {len(history)} messages")
            
            if not passed:
                conv_success = False
            
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": f"Response about {expected_kind}"})
        
        if conv_success:
            successful_conversations += 1
    
    print_header("MULTI-TURN SUMMARY", 2)
    success_rate = (successful_conversations / total_conversations * 100) if total_conversations else 0
    log_to_file(f"Conversation Success: {successful_conversations}/{total_conversations} ({success_rate:.1f}%)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    global LOG_FILE
    
    # Open log file
    LOG_FILE = open(LOG_FILENAME, "w", encoding="utf-8")
    
    try:
        log_to_file(f"{BLUE}{'='*90}")
        log_to_file("INNOVABOT CHATBOT - COMPREHENSIVE EVALUATION SUITE")
        log_to_file(f"Assignment 1 Part 2: Testing & Diagnostic Report")
        log_to_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_file(f"{'='*90}{RESET}\n")
        
        print_info("This evaluation focuses on assignment requirements and identifies specific issues")
        print_info("with retrieval quality, routing, and content completeness.")
        print_info(f"Results will be saved to: {LOG_FILENAME}\n")
        
        # Run all tests
        test_retrieval_quality()
        test_content_accuracy()
        test_auto_mode_routing()
        test_intent_classification()  # ✅ NEW TEST
        test_pii_guardrails()
        test_hallucination_prevention()
        test_edge_cases()
        test_multi_turn_conversations()
        
        # Final summary
        print_header("EVALUATION COMPLETE", 1)
        log_to_file(f"{GREEN}All tests completed. Review results above for areas needing improvement.{RESET}")
        log_to_file(f"\n{GREEN}Full results saved to: {LOG_FILENAME}{RESET}")
        
    finally:
        # Close log file
        if LOG_FILE is not None:
            LOG_FILE.close()
            print(f"\n{GREEN}✅ Evaluation results saved to: {LOG_FILENAME}{RESET}")

if __name__ == "__main__":
    main()
