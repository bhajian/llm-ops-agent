# app/rag_agent.py
"""
Retrieval-Augmented-Generation helper.

 • OPENAI  → ChatOpenAI + OpenAIEmbeddings
 • vLLM / llama.cpp / LM-Studio → OpenAI-compatible chat API + HF embeddings
 • LangChain < 0.2 *and* ≥ 0.2 are both supported
"""
from __future__ import annotations

import weaviate
from functools import lru_cache

# LangChain imports changed location in 0.2
try:  # ≥ 0.2
    from langchain_community.chains.retrieval_qa.base import RetrievalQA
except ImportError:  # < 0.2 fallback
    from langchain.chains.retrieval_qa import RetrievalQA

from langchain_community.vectorstores import Weaviate as WeaviateVS

from app.config import get_settings
from app.weaviate_utils import ensure_document_class
from app.llm import get_llm, get_embeddings


_cfg = get_settings()                     # single source of truth


# ───────────────── helpers ───────────────────────────────────
@lru_cache
def _client() -> weaviate.Client:
    """Singleton Weaviate client; ensures the Document class exists."""
    cli = weaviate.Client(_cfg["weaviate_url"])
    ensure_document_class(cli)
    return cli


# global singletons keep RAM low
_EMBED = get_embeddings()                 # OpenAI or HF model
_LLM   = get_llm(streaming=True)          # OpenAI or local vLLM/llama


# ───────────────── public factory ────────────────────────────
def get_cot_rag_chain() -> RetrievalQA:
    """
    Returns a RetrievalQA chain with:
      • Weaviate vector store
      • 4-doc similarity search
      • Streaming LLM (cloud or local)
    """
    vectordb  = WeaviateVS(_client(), "Document", "content", _EMBED)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=_LLM,
        retriever=retriever,
        chain_type="stuff",               # simplest combine mode
        return_source_documents=True,
    )

    # convenience attributes for router / tests
    qa_chain.retriever = retriever        # type: ignore[attr-defined]
    qa_chain.llm       = _LLM             # type: ignore[attr-defined]
    return qa_chain


# ─── legacy alias for old imports ────────────────────────────
def get_rag_chain():
    """Back-compat wrapper; prefer `get_cot_rag_chain`."""
    return get_cot_rag_chain()
