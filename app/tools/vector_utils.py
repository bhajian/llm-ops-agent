# app/tools/vector_utils.py

import os
from pathlib import Path
from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
# REMOVE these specific embedding imports, as we will use app.llm.get_embeddings
# from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.tools.weaviate_v4 import connect_weaviate, ensure_document_class
# ADD this import to use the centralized get_embeddings
from app.llm import get_embeddings # <--- ADD THIS LINE

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CLASS_NAME = "Document"
# REMOVE this local EMBEDDING_MODEL variable
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")  # or "huggingface"


def load_file(path: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    loader = PyPDFLoader(path) if ext == ".pdf" else TextLoader(path)
    return loader.load()


# REMOVE this local get_embeddings function
# def get_embeddings():
#     if EMBEDDING_MODEL == "openai":
#         return OpenAIEmbeddings()
#     else:
#         return HuggingFaceEmbeddings()


def ingest_file_to_weaviate(file_path: str) -> int:
    print(f"[INGEST] Loading file: {file_path}")
    docs = load_file(file_path)
    print(f"[INGEST] Loaded {len(docs)} raw docs")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    print(f"[INGEST] Split into {len(split_docs)} chunks")

    client = connect_weaviate(WEAVIATE_URL)
    client.connect()  # ✅ required to activate client in 4.15.x
    ensure_document_class(client, CLASS_NAME)
    collection = client.collections.get(CLASS_NAME)

    # Use the centralized get_embeddings from app.llm
    embedding_fn = get_embeddings().embed_documents # <--- MODIFIED LINE
    texts = [d.page_content for d in split_docs]
    vectors = embedding_fn(texts)

    print("[INGEST] Inserting into Weaviate using batch...")
    with collection.batch.fixed_size(len(vectors)) as batch:
        for doc, vector in zip(split_docs, vectors):
            batch.add_object(
                properties={
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", str(uuid4()))
                },
                vector=vector
            )

    print(f"[INGEST] ✅ Ingested {len(split_docs)} chunks to Weaviate.")
    return len(split_docs)
