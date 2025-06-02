# app/tools/vector_utils.py
"""
Utilities to ingest raw strings or files into Weaviate.
"""

import os, re
from pathlib import Path
from typing import Iterable, List

import weaviate
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Weaviate

# text-splitter import works for old & new LangChain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_community.text_splitter import RecursiveCharacterTextSplitter  # <0.2 fallback

from app.config import get_settings
from app.weaviate_utils import ensure_document_class


_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


# ───────── helpers ─────────
def _embeddings():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _client() -> weaviate.Client:
    cfg = get_settings()
    client = weaviate.Client(cfg["weaviate_url"])
    ensure_document_class(client)
    return client


def _db() -> Weaviate:
    return Weaviate(_client(), "Document", "content", _embeddings())


# ───────── public API ─────────
def ingest_texts(texts: Iterable[str]) -> int:
    docs: List[Document] = []
    for txt in texts:
        for chunk in _SPLITTER.split_text(txt):
            docs.append(Document(page_content=chunk))
    if docs:
        _db().add_documents(docs)
    return len(docs)


def ingest_files(paths: Iterable[str]) -> int:
    all_texts: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(path)

        loader_cls = PyPDFLoader if re.search(r"\.pdf$", path.name, re.I) else TextLoader
        docs = loader_cls(str(path)).load()
        all_texts.extend(d.page_content for d in docs)

    return ingest_texts(all_texts)
