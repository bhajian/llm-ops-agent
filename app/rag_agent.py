# app/rag_agent.py
import asyncio
from typing import List, Dict

import weaviate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate as WeaviateVS
from langchain.chains import create_qa_with_sources_chain

from app.config import get_settings
from app.weaviate_utils import ensure_document_class


# ─── helpers ───
def _client() -> weaviate.Client:
    cfg = get_settings()
    c = weaviate.Client(cfg["weaviate_url"])
    ensure_document_class(c)
    return c

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
_EMBED    = OpenAIEmbeddings()  # fallback to HF handled upstream

def get_rag_chain():
    """
    Back-compat for code that still does:
        from app.rag_agent import get_rag_chain
    It simply returns the new CoT-enabled chain.
    """
    return get_cot_rag_chain()

# ─── public: CoT-RAG chain ───
def get_cot_rag_chain() -> create_qa_with_sources_chain:
    vectordb = WeaviateVS(_client(), "Document", "content", _EMBED)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)
    return create_qa_with_sources_chain(llm_answer, retriever=retriever)


# ─── utility to run searches asynchronously ───
async def run_parallel_searches(queries: List[str], retriever):
    loop = asyncio.get_running_loop()
    coros = [loop.run_in_executor(None, retriever.get_relevant_documents, q)
             for q in queries]
    results = await asyncio.gather(*coros)
    flat = [d for sub in results for d in sub]
    return flat[:8]            # cap context size
