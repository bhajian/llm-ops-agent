# ui/pages/2_upload.py
import os, tempfile, streamlit as st
from client import ingest_file

st.set_page_config(page_title="LLM DevOps · Upload", page_icon="📄")
st.title("Upload & Ingest")

uploaded = st.file_uploader("PDF or TXT file", type=["pdf", "txt"])
if uploaded and st.button("Ingest"):
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded.name.split(".")[-1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        st.success(ingest_file(tmp_path))
    except Exception as e:
        st.error(f"Ingest failed: {e}")
    finally:
        os.remove(tmp_path)

st.caption("Uploaded docs are chunked, vectorised, and stored in Weaviate.")
