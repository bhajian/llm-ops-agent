# app/tools/vector_utils.py
"""
Utilities to ingest raw strings or files into Weaviate.
"""

import os
from pathlib import Path
from typing import Iterable, List

import weaviate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate

from app.config import get_settings
from app.weaviate_utils import ensure_document_class

# ---------------- config ----------------
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)


def _embeddings():
    return (
        OpenAIEmbeddings()
        if "OPENAI_API_KEY" in os.environ
        else HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )


def _client() -> weaviate.Client:
    cfg = get_settings()
    client = weaviate.Client(cfg["weaviate_url"])
    ensure_document_class(client)
    return client


def _db() -> Weaviate:
    return Weaviate(_client(), "Document", "content", _embeddings())


# ---------------- public helpers ----------------
def ingest_texts(texts: Iterable[str]) -> int:
    docs: List[dict] = []
    for txt in texts:
        for chunk in _TEXT_SPLITTER.split_text(txt):
            docs.append({"content": chunk})
    if docs:
        _db().add_documents(docs)
    return len(docs)


def ingest_files(paths: Iterable[str]) -> int:
    texts: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(path)

        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path))
        docs = loader.load()
        texts.extend(d.page_content for d in docs)
    return ingest_texts(texts)
