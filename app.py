import streamlit as st
from rag import RAGSystem
import time

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Netpin AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# CUSTOM DARK THEME
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.chat-container {
    max-width: 850px;
    margin: auto;
}
.stChatMessage {
    padding: 14px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# STREAMING FUNCTION (FIXED)
# -------------------------
def stream_markdown(text, placeholder):
    streamed = ""
    for char in text:
        streamed += char
        placeholder.markdown(streamed, unsafe_allow_html=True)
        time.sleep(0.003)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("🤖 Netpin AI")

    st.markdown("### ⚙️ Controls")
    show_sources = st.toggle("📚 Show Sources", True)
    debug_mode = st.toggle("🧠 Debug Mode", False)

    st.markdown("---")

    st.markdown("### 💡 Try asking")
    example_queries = [
        "How to setup Netpin?",
        "Why did my deployment fail?",
        "Explain Infrastructure Debt Index",
        "How to fix CrashLoopBackOff?"
    ]

    for q in example_queries:
        if st.button(q):
            st.session_state.example_prompt = q

    st.markdown("---")

    st.markdown("### ℹ️ System Info")
    st.write("Model: Llama 3 (HF API)")
    st.write("Retrieval: Hybrid (FAISS + BM25)")
    st.write("Reranker: CrossEncoder")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.rag.chat_history.clear()
        st.rerun()

# -------------------------
# HEADER
# -------------------------
st.title("💬 Netpin AI Assistant")
st.caption("Your intelligent DevOps companion 🚀")

# -------------------------
# INIT SESSION
# -------------------------
if "rag" not in st.session_state:
    with st.spinner("🔄 Initializing AI..."):
        st.session_state.rag = RAGSystem()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "example_prompt" not in st.session_state:
    st.session_state.example_prompt = None

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"], unsafe_allow_html=True)

# -------------------------
# INPUT
# -------------------------
prompt = st.chat_input("Ask anything about Netpin...")

# Handle example click
if st.session_state.example_prompt:
    prompt = st.session_state.example_prompt
    st.session_state.example_prompt = None

# -------------------------
# MAIN CHAT LOGIC
# -------------------------
if prompt:
    # USER MESSAGE
    st.chat_message("user", avatar="👤").markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # -------------------------
    # AI RESPONSE
    # -------------------------
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()

        with st.spinner("Thinking..."):
            result = st.session_state.rag.ask(prompt)

        answer = result["answer"]
        sources = result["sources"]

        # -------------------------
        # SMART STREAMING (FIXED)
        # -------------------------
        if len(answer) < 1500:
            stream_markdown(answer, placeholder)
        else:
            # fallback for long responses
            placeholder.markdown(answer, unsafe_allow_html=True)

        # -------------------------
        # SOURCES PANEL
        # -------------------------
        if show_sources and sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

        # -------------------------
        # FEEDBACK BUTTONS
        # -------------------------
        col1, col2 = st.columns(2)
        if col1.button("👍 Helpful"):
            st.toast("Thanks for your feedback!")
        if col2.button("👎 Not Helpful"):
            st.toast("Got it! We'll improve.")

        # -------------------------
        # DEBUG MODE
        # -------------------------
        if debug_mode:
            st.markdown("### 🧠 Debug Info")
            st.json(result)

    # SAVE RESPONSE
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })