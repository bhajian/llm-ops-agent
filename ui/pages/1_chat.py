# pages/1_chat.py

import streamlit as st
from client import stream_chat
from sidebar import render_chat_sidebar  # Ensure sidebar.py exists and works

st.set_page_config(page_title="LLM DevOps · Chat", page_icon="💬")

# ─── Sidebar UI ───────────────────────────────────────────────
with st.sidebar:
    render_chat_sidebar()

# ─── Current chat session ID ──────────────────────────────────
chat_id = st.session_state.get("chat_id")
if not chat_id:
    st.warning("No chat session selected. Please select or create a new chat from the sidebar.")
    st.stop()

# ─── Per-chat session key ─────────────────────────────────────
key = f"messages_{chat_id}"

# ─── Initialize chat history if first time ────────────────────
if key not in st.session_state:
    st.session_state[key] = []

# ─── Chat display container ───────────────────────────────────
chat_box = st.container()

for msg in st.session_state[key]:
    with chat_box.chat_message(msg["role"]):  # Only accepts "user" or "assistant"
        st.markdown(msg["content"])

# ─── Input prompt box ─────────────────────────────────────────
prompt = st.chat_input("Ask me anything...")

if prompt:
    # 1. Display user prompt
    st.session_state[key].append({"role": "user", "content": prompt})
    with chat_box.chat_message("user"):
        st.markdown(prompt)

    # 2. Stream assistant response
    with chat_box.chat_message("assistant"):
        full_response = ""
        response_placeholder = st.empty()

        with st.status(label="Thinking...", expanded=True, state="running") as status:
            try:
                for token in stream_chat(prompt, chat_id):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")  # Blinking cursor

                response_placeholder.markdown(full_response)
                status.update(label="Response generated!", state="complete", expanded=False)

            except Exception as e:
                full_response = f"❌ Error: {e}"
                response_placeholder.markdown(full_response)
                status.update(label="An error occurred!", state="error", expanded=True)

    # 3. Save assistant reply to session
    st.session_state[key].append({"role": "assistant", "content": full_response})
