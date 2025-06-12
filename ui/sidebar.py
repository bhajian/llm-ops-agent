"""
ui/sidebar.py
─────────────
Sidebar widget: list, select, delete conversations.
"""

import streamlit as st
import uuid

import client


def render_chat_sidebar() -> None:
    st.sidebar.title("💬 Chats")

    def _refresh():
        st.session_state.chats = client.list_history()
        # Ensure that if there are no chats, a new one is created.
        if not st.session_state.chats:
            st.session_state.chat_id = uuid.uuid4().hex
            st.session_state.chats = [st.session_state.chat_id] # Initialize with the new chat ID
        elif "chat_id" not in st.session_state or st.session_state.chat_id not in st.session_state.chats:
            # If chat_id is not set or not in existing chats, default to the first one.
            st.session_state.chat_id = st.session_state.chats[0]


    if "chats" not in st.session_state:
        _refresh()
    # Ensure chat_id is set even on initial load if not already.
    if "chat_id" not in st.session_state:
        _refresh() # This will ensure chat_id is set if it's missing on initial load


    # Determine the initial index for the radio button
    try:
        initial_index = st.session_state.chats.index(st.session_state.chat_id)
    except ValueError:
        # This can happen if st.session_state.chat_id is not in the list of chats
        # (e.g., after a delete). In this case, default to the first chat if available,
        # or re-initialize.
        if st.session_state.chats:
            initial_index = 0
            st.session_state.chat_id = st.session_state.chats[0]
        else:
            # This case should ideally be handled by _refresh(), but as a fallback
            _refresh() # Re-call to ensure a chat_id is generated and chats list is updated
            initial_index = 0 # Default to the newly created chat
            
    sel = st.sidebar.radio(
        "Conversations",
        st.session_state.chats,
        index=initial_index,
        key="conversation_selector" # Add a unique key to prevent warning
    )
    st.session_state.chat_id = sel

    if st.sidebar.button("➕ New Chat"):
        st.session_state.chat_id = uuid.uuid4().hex
        st.session_state.chats.insert(0, st.session_state.chat_id) # Add new chat to the top
        st.session_state.messages = {} # Clear messages for the new chat
        st.rerun() # Rerun to update the UI with the new chat

    if st.sidebar.button("🗑️ Delete Current Chat"):
        client.delete_history(st.session_state.chat_id)
        # Remove the deleted chat from the session state list
        if st.session_state.chat_id in st.session_state.chats:
            st.session_state.chats.remove(st.session_state.chat_id)
        
        # If there are still chats, select the first one, otherwise create a new one
        if st.session_state.chats:
            st.session_state.chat_id = st.session_state.chats[0]
        else:
            st.session_state.chat_id = uuid.uuid4().hex
            st.session_state.chats = [st.session_state.chat_id] # Ensure chats list is not empty
        
        st.session_state.messages = {} # Clear messages for the (newly selected/created) chat
        _refresh() # Refresh the list from the server
        st.rerun() # Rerun to update the UI with the new chat
        