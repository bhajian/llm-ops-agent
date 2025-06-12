import streamlit as st
from client import stream_chat, get_history # Import get_history
from sidebar import render_chat_sidebar

st.set_page_config(page_title="LLM Chat", page_icon="💬")

# Sidebar
with st.sidebar:
    render_chat_sidebar()

chat_id = st.session_state.chat_id
history_key = f"messages_{chat_id}"

# Load history if not already loaded for the current chat_id
if history_key not in st.session_state:
    st.session_state[history_key] = get_history(chat_id) # Load from backend


# Show previous messages
for msg in st.session_state[history_key]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input box
if user_msg := st.chat_input("Say something…"):
    # show user message
    st.chat_message("user").write(user_msg)
    st.session_state[history_key].append({"role": "user", "content": user_msg})

    # stream assistant response
    with st.chat_message("assistant"):
        full_resp = ""
        placeholder = st.empty()
        for line in stream_chat(chat_id, user_msg):
            if line.startswith(b"data: "):
                chunk = line[6:].decode().strip()
                if chunk == "[DONE]":
                    break
                if chunk.startswith("[ERROR]"): # Handle server-side errors
                    placeholder.error(chunk)
                    full_resp = "" # Don't save error as assistant message
                    break
                full_resp += chunk
                placeholder.markdown(full_resp + "▌")
        placeholder.markdown(full_resp or "_(no response)_")

    # save assistant reply if not an error
    if full_resp: # Only save if there was a valid response
        st.session_state[history_key].append({"role": "assistant", "content": full_resp})
        