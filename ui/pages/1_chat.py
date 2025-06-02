import streamlit as st
from client import stream_chat
from sidebar import render_chat_sidebar

st.set_page_config(page_title="LLM DevOps · Chat", page_icon="💬")

# ─── Sidebar UI ───────────────────────────────────────────────
with st.sidebar:
    render_chat_sidebar()

# ─── Current chat session ID ──────────────────────────────────
chat_id = st.session_state.get("chat_id")
if not chat_id:
    st.warning("No chat session selected.")
    st.stop()

# Use per-chat message key
key = f"messages_{chat_id}"

# ─── Initialize chat history if missing ───────────────────────
if key not in st.session_state:
    st.session_state[key] = []

# ─── Display Chat History ─────────────────────────────────────
chat_box = st.container()
for msg in st.session_state[key]:
    with chat_box.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input & Streaming Response ───────────────────────────────
prompt = st.chat_input("Ask me anything...")
if prompt:
    st.session_state[key].append({"role": "user", "content": prompt})
    with chat_box.chat_message("user"):
        st.markdown(prompt)

    with chat_box.chat_message("assistant"):
        full_response = ""
        block = st.empty()
        try:
            for token in stream_chat(prompt, chat_id):
                full_response += token
                block.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"❌ {e}"
        block.markdown(full_response)

    st.session_state[key].append({"role": "assistant", "content": full_response})
