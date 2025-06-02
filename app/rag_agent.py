# app/rag_agent.py

"""
Backend-agnostic RAG module.

 • OPENAI  → ChatOpenAI + OpenAIEmbeddings
 • vLLM / llama.cpp → OpenAI-compatible chat + HF embeddings
 • Works with old (<0.2) *and* new (≥0.2) LangChain trees.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Iterable, List

from langchain_weaviate.vectorstores import WeaviateVectorStore

# -------- robust RetrievalQA import (handles many LC versions) ----------
RetrievalQA = None
for _path in (
    "langchain_community.chains.retrieval_qa.base",
    "langchain_community.chains.retrieval_qa",
    "langchain.chains.retrieval_qa.base",
    "langchain.chains.retrieval_qa",
):
    try:
        module = __import__(_path, fromlist=["RetrievalQA"])
        RetrievalQA = module.RetrievalQA  # type: ignore
        break
    except (ImportError, AttributeError):
        continue
if RetrievalQA is None:
    raise ImportError("Could not locate `RetrievalQA` in installed LangChain")

from app.config import get_settings
from app.llm import get_llm, get_embeddings
from app.tools.weaviate_v4 import connect_weaviate, ensure_document_class

_cfg = get_settings()

# ───────────────── helpers ───────────────────────────────────────────────
@lru_cache
def _client():
    """Singleton Weaviate v4 client with schema ensured and connected."""
    client = connect_weaviate(_cfg["weaviate_url"])
    client.connect()
    ensure_document_class(client)
    return client

@lru_cache
def _embedding():
    return get_embeddings()

# ───────────────── public factory ────────────────────────────────────────
def get_cot_rag_chain(streaming: bool = False) -> RetrievalQA:
    """
    Build & return a RetrievalQA chain:
      • Weaviate vector store
      • k=4 similarity search
      • LLM (streaming or not)
    """
    client = _client()

    vectordb = WeaviateVectorStore(
        client=client,
        index_name="Document",
        text_key="text",
        embedding=_embedding(),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(streaming=streaming)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        input_key="query"
    )

# ───────────────── util: parallel retrieval ─────────────────────────────
async def run_parallel_searches(
    queries: Iterable[str],
    retriever,
) -> List:
    """
    Fire retriever.get_relevant_documents concurrently.
    Returns a flat list of docs.
    """
    tasks = []
    for q in queries:
        if hasattr(retriever, "aget_relevant_documents"):
            tasks.append(retriever.aget_relevant_documents(q))  # async
        else:
            tasks.append(asyncio.to_thread(retriever.get_relevant_documents, q))
    results = await asyncio.gather(*tasks, return_exceptions=False)

    docs = []
    for batch in results:
        docs.extend(batch)
    return docs

# ─── legacy alias for old code ───────────────────────────────────────────
def get_rag_chain():
    return get_cot_rag_chain()
