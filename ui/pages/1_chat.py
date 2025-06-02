# ui/pages/1_chat.py
import uuid, streamlit as st
from client import stream_chat

st.set_page_config(page_title="LLM DevOps · Chat", page_icon="💬")

# session state
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# sidebar
with st.sidebar:
    st.header("Chat settings")
    st.selectbox("Model", ["openai-gpt-4o", "openai-gpt-4o-mini"], key="model")
    if st.button("🔄 Clear chat"):
        st.session_state.messages.clear()
        st.rerun()

# history
chat_box = st.container()
for msg in st.session_state.messages:
    with chat_box.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
prompt = st.chat_input("Ask me anything…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_box.chat_message("user"):
        st.markdown(prompt)

    holder, partial = chat_box.chat_message("assistant").empty(), ""
    try:
        for token in stream_chat(prompt, st.session_state.chat_id):
            partial += token
            holder.markdown(partial + "▌")
    except Exception as e:
        partial = f"❌ {e}"
    holder.markdown(partial)
    st.session_state.messages.append({"role": "assistant", "content": partial})
