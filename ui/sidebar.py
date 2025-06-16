import uuid
import streamlit as st
from client import list_chats, delete_chat, load_history, init_chat

def render_chat_sidebar():
    st.header("💬 Chat Sessions")

    # Initialize chat session list
    if "session_chats" not in st.session_state:
        st.session_state.session_chats = set(list_chats())

    # ➕ New Chat button
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        init_chat(new_id, "Start chat")
        st.session_state.session_chats.add(new_id)
        st.session_state.chat_id = new_id
        st.session_state[f"messages_{new_id}"] = load_history(new_id)
        st.rerun()

    st.markdown("---")

    # Current chat ID
    chat_id = st.session_state.get("chat_id")
    if not chat_id:
        chat_id = str(uuid.uuid4())
        init_chat(chat_id, "Start chat")
        st.session_state.chat_id = chat_id
        st.session_state.session_chats.add(chat_id)
        st.session_state[f"messages_{chat_id}"] = load_history(chat_id)

    # Combine Redis and local session IDs (filtering None)
    redis_chats = list_chats()
    known_chats = sorted(set(filter(None, redis_chats + list(st.session_state.session_chats))))

    # Render chat list
    for cid in known_chats:
        is_active = cid == chat_id
        label = f"🟢 {cid[:8]}" if is_active else f"⚪️ {cid[:8]}"
        col1, col2 = st.columns([0.75, 0.25])

        with col1:
            if st.button(label, key=f"switch-{cid}"):
                st.session_state.chat_id = cid
                if f"messages_{cid}" not in st.session_state:
                    st.session_state[f"messages_{cid}"] = load_history(cid)
                st.rerun()

        with col2:
            if st.button("🗑", key=f"delete-{cid}"):
                delete_chat(cid)
                st.session_state.session_chats.discard(cid)

                if cid == chat_id:
                    # Auto-switch to a new chat
                    new_id = str(uuid.uuid4())
                    init_chat(new_id, "Start chat")
                    st.session_state.chat_id = new_id
                    st.session_state.session_chats.add(new_id)
                    st.session_state[f"messages_{new_id}"] = load_history(new_id)
                st.rerun()

    st.markdown("---")
    st.selectbox("🧠 Model", ["vllm/llama-3.1-8B", "llama4-scout"], key="model")
