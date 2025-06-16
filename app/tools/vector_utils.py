# app/tools/vector_utils.py
"""
Light-weight helper that turns a local PDF/TXT file into
vector-store chunks stored in Weaviate (v3 client).

Steps
-----
1.  Load the file (PDF → PyPDFLoader, everything else → TextLoader)
2.  Split into ~1 kB chunks with overlap
3.  Embed each chunk with `get_embeddings()`   (OpenAI or HF – same as RAG)
4.  Upsert into the `Document` class inside Weaviate
"""

from __future__ import annotations
import os, uuid
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.weaviate import Weaviate as WeaviateVectorStore
from weaviate import Client as WeaviateClient

from app.llm import get_embeddings


# ────────────────────────── Weaviate connection ──────────────────────────
def _connect_weaviate() -> WeaviateClient:
    """
    Returns a v3 weaviate.Client instance and creates the `Document` class
    (vectorizer = NONE) if it doesn't exist yet.
    """
    url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    client = WeaviateClient(url)

    schema = client.schema.get() or {}
    classes = {c["class"] for c in schema.get("classes", [])}
    if "Document" not in classes:
        client.schema.create_class(
            {
                "class": "Document",
                "description": "Generic text chunks",
                "vectorizer": "none",  # we provide our own embeddings
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "source", "dataType": ["text"]},
                ],
            }
        )
    return client


# ───────────────────── file → list[Document] helper ──────────────────────
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)


def _load_and_split(path: str | Path) -> List[Document]:
    """
    Loads *any* .pdf, .txt, .md, … into LangChain Documents and splits.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    return _TEXT_SPLITTER.split_documents(docs)


# ───────────────────────── public ingest helper ──────────────────────────
def ingest_file_to_weaviate(path: str | Path) -> int:
    """
    Main function used by the /ingest endpoint.
    Returns *number of chunks* inserted.
    """
    docs = _load_and_split(path)
    if not docs:
        return 0

    # enrich with a simple `source` meta-field
    for d in docs:
        d.metadata["source"] = Path(path).name

    embed = get_embeddings()                     # OpenAI or HF
    client = _connect_weaviate()

    vs = WeaviateVectorStore(
        client=client,
        embedding=embed,            # => vectors are supplied client-side
        index_name="Document",
        text_key="text",            # where the raw chunk is stored
        by_text=False,              # store the vectors directly
    )

    vs.add_documents(docs)
    return len(docs)
