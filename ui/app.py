# ui/app.py
import os, uuid, requests, streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://agent-server:8000")
AUTH_TOKEN  = os.getenv("AUTH_TOKEN",  "supersecrettoken")

st.set_page_config(page_title="LLM DevOps Chat", page_icon="🤖", layout="wide")
st.title("🤖 LLM DevOps Chat")

# ---------- session-state ----------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []            # {"role": ..., "content": ...}

# ---------- render history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- input ----------
prompt = st.chat_input("Ask me anything…")
if prompt:
    # show user bubble first
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # pre-create assistant bubble we’ll progressively fill
    assistant_holder = st.chat_message("assistant").empty()

    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    payload = {"query": prompt, "chat_id": st.session_state.chat_id}

    # ---------- streaming call ----------
    try:
        with requests.post(f"{BACKEND_URL}/chat/stream",
                           json=payload, headers=headers,
                           stream=True, timeout=120) as r:
            r.raise_for_status()

            partial = ""
            # iterate over *raw* bytes – chunk_size=None → get tokens quickly
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:                   # filter out keep-alives
                    partial += chunk
                    assistant_holder.markdown(partial + "▌")  # cursor
    except Exception as e:
        partial = f"❌ Error: {e}"
        assistant_holder.markdown(partial)

    # remove cursor, save final text
    assistant_holder.markdown(partial)
    st.session_state.messages.append({"role": "assistant", "content": partial})
