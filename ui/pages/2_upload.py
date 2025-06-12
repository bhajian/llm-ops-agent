import streamlit as st
from client import ingest_file

st.set_page_config(page_title="Doc Upload", page_icon="📄")
st.title("📄 Upload & Ingest into RAG")

file = st.file_uploader("Choose a PDF or TXT", type=["pdf", "txt"])
if file and st.button("📤 Ingest"):
    with st.spinner("Uploading…"):
        st.success(ingest_file(file))
