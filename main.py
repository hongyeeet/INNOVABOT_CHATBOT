import streamlit as st

from src.agents import route_query, classify_intent_and_route
from src.llm_client import generate_response
from src.prompts import build_pii_refusal_message, build_unsafe_content_message
from src.guardrails import contains_pii, is_content_unsafe


# ---------- Page config ----------
st.set_page_config(
    page_title="INNOVABOT Chatbot",
    page_icon="💬",
    layout="wide",
)


# ---------- Theming ----------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # ← CHANGED: Default to dark


def apply_theme():
    base = st.session_state.theme
    if base == "dark":
        st._config.set_option("theme.base", "dark")
        st._config.set_option("theme.primaryColor", "#4F9DDE")
        st._config.set_option("theme.backgroundColor", "#0E1117")
        st._config.set_option("theme.secondaryBackgroundColor", "#161B22")
        st._config.set_option("theme.textColor", "#FAFAFA")
    else:
        st._config.set_option("theme.base", "light")
        st._config.set_option("theme.primaryColor", "#2563EB")
        st._config.set_option("theme.backgroundColor", "#FFFFFF")
        st._config.set_option("theme.secondaryBackgroundColor", "#F3F4F6")
        st._config.set_option("theme.textColor", "#111827")


apply_theme()


# ---------- Global CSS ----------
st.markdown(
    """
    <style>
    /* (no suggested-question styling needed) */
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


if "mode" not in st.session_state:
    st.session_state.mode = "Auto"


if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3


if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False


# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("### Chatbot controls")

    mode = st.radio(
        "Mode",
        options=["Brochure only", "Wikipedia", "Auto"],
        index=["Brochure only", "Wikipedia", "Auto"].index(st.session_state.mode),
    )
    st.session_state.mode = mode

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.temperature),
        step=0.05,
        help="Controls randomness of responses.",
    )
    st.session_state.temperature = temperature

    show_sources = st.toggle(
        "Show sources",
        value=st.session_state.show_sources,
        help="Include source attributions in responses.",
    )
    st.session_state.show_sources = show_sources

    theme_choice = st.toggle(
        "Dark theme",
        value=(st.session_state.theme == "dark"),
        help="Toggle between light and dark themes.",
    )
    new_theme = "dark" if theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        apply_theme()
        st.rerun()


# ---------- Header ----------
st.markdown("## INNOVABOT Chatbot")
st.caption(
    f"Multimodel assistant · Mode: **{st.session_state.mode}** · "
    f"Temperature: **{st.session_state.temperature:.2f}**"
)
st.markdown("---")


# ---------- Conversation history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and st.session_state.show_sources and msg["role"] == "assistant":
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")


# Optional "thinking…" bubble when waiting for the last response
if st.session_state.is_thinking:
    with st.chat_message("assistant"):
        st.markdown("_Thinking…_")


# ---------- Chat input ----------
prompt = st.chat_input("Ask a question about EngagePro or general topics...")


# ---------- Handle submit ----------
if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.is_thinking = True
    st.rerun()


# ---------- If we are in thinking state, compute the answer now ----------
if st.session_state.is_thinking:
    last_user = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            last_user = m["content"]
            break

    if last_user:
        # ✅ STEP 1: Check PII FIRST (before any classification)
        if contains_pii(last_user):
            reply_text = build_pii_refusal_message()
            sources = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply_text,
                "sources": sources,
            })
            st.session_state.is_thinking = False
            st.rerun()
        
        # ✅ STEP 1.5: Check content safety SECOND (before classification)
        print("="*50)
        print(f"🔍 SAFETY CHECK STARTING")
        print(f"Query to check: '{last_user}'")
        print("="*50)

        try:
            is_unsafe, reason = is_content_unsafe(last_user)
            print(f"✅ Safety check completed")
            print(f"   is_unsafe = {is_unsafe}")
            print(f"   reason = '{reason}'")
        except Exception as e:
            print(f"❌ ERROR in safety check: {e}")
            import traceback
            traceback.print_exc()
            is_unsafe = False
            reason = ""

        print("="*50)

        if is_unsafe:
            print(f"🚫 BLOCKING unsafe content: {reason}")
            reply_text = build_unsafe_content_message()
            sources = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply_text,
                "sources": sources,
            })
            st.session_state.is_thinking = False
            st.rerun()
        else:
            print(f"✅ Content is safe, continuing to classification")

        
        # ✅ STEP 2: Combined intent + route classification (ONE API call!)
        history_for_routing = [
            m for m in st.session_state.messages if m["role"] in ("user", "assistant")
        ]
        
        classification = classify_intent_and_route(last_user, history_for_routing)
        
        # ✅ STEP 3: Handle greetings (no RAG needed)
        if classification == 'greeting':
            reply_text = (
                "Hello! How can I assist you today? "
                "If you have any questions about EngagePro or our "
                "products and services, feel free to ask!"
            )
            sources = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply_text,
                "sources": sources,
            })
            st.session_state.is_thinking = False
            st.rerun()
        
        # ✅ STEP 4: Handle statements (no RAG needed)
        elif classification == 'statement':
            reply_text = (
                "I appreciate your feedback! "
                "Is there anything else I can help you with?"
            )
            sources = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply_text,
                "sources": sources,
            })
            st.session_state.is_thinking = False
            st.rerun()
        
        # ✅ STEP 5: Handle questions (brochure or wiki)
        elif classification in ['question_brochure', 'question_wiki']:
            # Determine mode based on classification and user settings
            if st.session_state.mode == "Auto":
                # Use classifier's decision
                mode = "Brochure only" if classification == 'question_brochure' else "Wikipedia"
            else:
                # Respect user's manual override
                mode = st.session_state.mode
            
            route_result = route_query(
                query=last_user,
                mode=mode,
                history=history_for_routing,
            )
            
            contexts = route_result["contexts"]
            sources_used = route_result["sources_used"]
            kind = route_result["kind"]
            sanitized_query = route_result.get("sanitized_query", last_user)
            
            history_for_llm = [
                m for m in st.session_state.messages if m["role"] in ("user", "assistant")
            ]
            
            answer_text, usage = generate_response(
                kind=kind,
                user_query=sanitized_query,
                contexts=contexts,
                history=history_for_llm,
                temperature=st.session_state.temperature,
            )
            
            reply_text = answer_text
            sources = sources_used
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply_text,
                "sources": sources,
            })
            
            st.session_state.is_thinking = False
            st.rerun()
