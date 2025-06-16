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

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


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
def ingest_file_to_weaviate(path: Union[str, Path]) -> int:
    """
    Load → split → embed → upsert.
    Returns the number of chunks written, *0* on empty files.
    Raises *RuntimeError* on any unexpected failure (so FastAPI turns it into 500).
    """
    path = Path(path)
    log.info("📥  Ingest started: %s", path)

    # ------------------------------------------------------------------ 1) load + split
    try:
        docs: List = _load_and_split(path)
        log.info("📝  %s → %d chunks", path.name, len(docs))
    except Exception as err:
        log.exception("❌  Failed during load/split")
        raise RuntimeError(f"load/split failed for {path}: {err}") from err

    if not docs:                     # empty PDF / TXT
        log.warning("⚠️  No text extracted from %s – skipping", path)
        return 0

    for d in docs:                   # tiny enrichment
        d.metadata["source"] = path.name

    # ------------------------------------------------------------------ 2) embeddings client
    try:
        embed = get_embeddings()
    except Exception as err:
        log.exception("❌  Could not initialise embeddings backend")
        raise RuntimeError(f"embedding backend error: {err}") from err

    # ------------------------------------------------------------------ 3) weaviate connection
    try:
        client = _connect_weaviate()
    except Exception as err:
        log.exception("❌  Could not connect to Weaviate")
        raise RuntimeError(f"weaviate connection error: {err}") from err

    # ------------------------------------------------------------------ 4) upsert
    try:
        vs = WeaviateVectorStore(
            client=client,
            embedding=embed,
            index_name="Document",
            text_key="text",
            by_text=False,
        )
        vs.add_documents(docs)
        log.info("✅  Successfully ingested %d chunks from %s", len(docs), path)
        return len(docs)

    except Exception as err:
        log.exception("❌  Failed during upsert")
        raise RuntimeError(f"vector-store upsert failed: {err}") from err
