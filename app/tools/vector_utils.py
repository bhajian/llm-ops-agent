# app/tools/vector_utils.py

import os
from pathlib import Path
from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local imports (ADJUSTED PATHS)
from app.tools.weaviate_v4 import connect_weaviate, ensure_document_class
from app.llm import get_embeddings # llm is in common directory


# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CLASS_NAME = "Document"


def load_file(path: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    loader = PyPDFLoader(path) if ext == ".pdf" else TextLoader(path)
    return loader.load()


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
    embedding_fn = get_embeddings().embed_documents
    texts = [d.page_content for d in split_docs]
    vectors = embedding_fn(texts)

    print("[INGEST] Inserting into Weaviate using batch...")
    with collection.batch.fixed_size(len(vectors)) as batch:
        for doc, vector in zip(split_docs, vectors):
            batch.add_object(
                properties={
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                },
                vector=vector,
            )
    
    print(f"[INGEST] Successfully ingested {len(split_docs)} chunks.")
    return len(split_docs)

