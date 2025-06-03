import streamlit as st
from client import stream_chat
from sidebar import render_chat_sidebar # Assuming sidebar.py is in the same directory

st.set_page_config(page_title="LLM DevOps · Chat", page_icon="💬")

# ─── Sidebar UI ───────────────────────────────────────────────
with st.sidebar:
    render_chat_sidebar()

# ─── Current chat session ID ──────────────────────────────────
chat_id = st.session_state.get("chat_id")
if not chat_id:
    st.warning("No chat session selected. Please select or create a new chat from the sidebar.")
    st.stop()

# Use per-chat message key for session state
key = f"messages_{chat_id}"

# ─── Initialize chat history if missing ───────────────────────
if key not in st.session_state:
    st.session_state[key] = []

# ─── Display Chat History ─────────────────────────────────────
# We use a container to ensure messages are always rendered in the same place
chat_box = st.container()

for msg in st.session_state[key]:
    # Explicitly setting is_user for clear left/right alignment
    if msg["role"] == "user":
        with chat_box.chat_message("user", is_user=True): # Aligns to the right
            st.markdown(msg["content"])
    else: # role is "assistant"
        with chat_box.chat_message("assistant", is_user=False): # Aligns to the left
            st.markdown(msg["content"])

# ─── Input & Streaming Response ───────────────────────────────
prompt = st.chat_input("Ask me anything...")

if prompt:
    # 1. Add user prompt to chat history and display
    st.session_state[key].append({"role": "user", "content": prompt})
    with chat_box.chat_message("user", is_user=True):
        st.markdown(prompt)

    # 2. Display thinking animation and stream AI response
    with chat_box.chat_message("assistant", is_user=False):
        full_response = ""
        # Create an empty block to update with streaming tokens
        response_placeholder = st.empty()
        
        # Use st.status for the thinking animation
        with st.status("Thinking...", expanded=True, state="running") as status_container:
            try:
                # Stream the response from the backend
                for token in stream_chat(prompt, chat_id):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌") # Add a blinking cursor for typing effect
                
                # After streaming, remove the cursor and display the final response
                response_placeholder.markdown(full_response)
                
                # Update the status container to "complete" and optionally collapse it
                status_container.update(label="Response generated!", state="complete", expanded=False)
            
            except Exception as e:
                # Handle errors during streaming
                full_response = f"❌ Error: {e}"
                response_placeholder.markdown(full_response)
                status_container.update(label="An error occurred!", state="error", expanded=True) # Keep expanded to show error

    # 3. Append AI response to chat history
    st.session_state[key].append({"role": "assistant", "content": full_response)