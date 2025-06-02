import uuid
import streamlit as st
from client import stream_chat, load_history
from sidebar import render_chat_sidebar  # 👈 you modularized this

st.set_page_config(page_title="LLM DevOps · Chat", page_icon="💬")

# ─── Session State Init ───────────────────────────────────────
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

chat_id = st.session_state.chat_id

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    render_chat_sidebar()  # 👈 pulls in the full chat management UI

# ─── Load history only if missing ─────────────────────────────
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = load_history(chat_id)

# ─── Chat UI Display ──────────────────────────────────────────
chat_box = st.container()
for msg in st.session_state.messages:
    with chat_box.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input Prompt & Streamed Output ───────────────────────────
prompt = st.chat_input("Ask me anything…")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_box.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with chat_box.chat_message("assistant"):
        response_block = st.empty()
        full_response = ""
        try:
            for token in stream_chat(prompt, chat_id):
                full_response += token
                response_block.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"❌ {e}"
        response_block.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
