# ui/sidebar.py
import uuid, streamlit as st
from client import list_chats, delete_chat, load_history

def render_chat_sidebar():
    st.header("🧠 Chat Sessions")

    chat_id = st.session_state.get("chat_id", str(uuid.uuid4()))
    st.session_state.chat_id = chat_id

    all_chats = list_chats()
    if chat_id not in all_chats:
        all_chats.append(chat_id)

    selected = st.selectbox("🗂 Select chat", sorted(all_chats), index=sorted(all_chats).index(chat_id))
    if selected != chat_id:
        st.session_state.chat_id = selected
        st.session_state.messages = load_history(selected)
        st.rerun()

    if st.button("➕ New Chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑 Delete This Chat"):
        delete_chat(chat_id)
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.selectbox("🧠 Model", ["openai-gpt-4o", "openai-gpt-4o-mini"], key="model")
