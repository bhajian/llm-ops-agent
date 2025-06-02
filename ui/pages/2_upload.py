import streamlit as st
from client import ingest_file

st.set_page_config(page_title="LLM DevOps · Upload", page_icon="📄")
st.title("📄 Upload & Ingest")

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
if uploaded and st.button("📤 Ingest into RAG"):
    with st.spinner("Uploading and processing..."):
        try:
            msg = ingest_file(uploaded)
            if "fail" in msg.lower():
                st.error(msg)
            else:
                st.success(msg)
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

st.caption("Uploaded docs are chunked, vectorized, and stored in Weaviate.")
