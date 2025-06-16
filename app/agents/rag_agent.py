# app/agents/rag_agent.py
"""
RAG helper that:
  • ensures the "Document" class exists
  • embeds every user query locally
  • issues a Weaviate `nearVector` search (so no nearText errors)
  • exposes a Runnable with `.ainvoke()` for the planner/reasoner
"""
from __future__ import annotations
from typing import List
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.tools.weaviate_v4 import connect_weaviate, ensure_document_class
from app.llm import get_embeddings

# ────────────── bootstrap schema & vector-store ──────────────
_client = connect_weaviate()
ensure_document_class(_client, class_name="Document", text_key="text")

_embeddings = get_embeddings()

_VECSTORE = Weaviate(
    client=_client,
    index_name="Document",
    text_key="text",
    embedding=_embeddings,          # still used for ingest
)

def _vector_search(query: str, k: int = 4) -> List[Document]:
    """
    Embed the query → nearVector search (avoids nearText).
    """
    vec = _embeddings.embed_query(query)
    return _VECSTORE.similarity_search_by_vector(vec, k=k)

# expose as Runnable so `await _RETRIEVER.ainvoke(q)` just works
_RETRIEVER = RunnableLambda(lambda q: _vector_search(q, k=4))

# helper for reasoner
def _format_docs(docs: List[Document], max_chars: int = 8_000) -> str:
    out, tot = [], 0
    for d in docs:
        seg = d.page_content.strip()
        if tot + len(seg) > max_chars:
            break
        out.append(seg)
        tot += len(seg)
    return "\n\n".join(out)
