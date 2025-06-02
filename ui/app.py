# ui/app.py
import streamlit as st

st.set_page_config(page_title="LLM DevOps · Home", page_icon="🤖")

st.title("🤖 LLM DevOps Assistant")
st.markdown(
    """
Welcome!  
Use the sidebar to:

* **💬 Chat** – Ask questions about market data, Kubernetes, or anything else.
* **📄 Upload** – Add PDFs / text docs to the knowledge-base (Weaviate RAG).
"""
)
st.info("Tip: the chat remembers your conversation history until you clear it.")
