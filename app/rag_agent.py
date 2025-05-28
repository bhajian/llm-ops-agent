# app/rag_agent.py
import os
import weaviate
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import Weaviate
from langchain.chains import RetrievalQA

from app.llm import get_llm
from app.config import get_settings
from app.weaviate_utils import ensure_document_class


def _client() -> weaviate.Client:
    cfg = get_settings()
    client = weaviate.Client(cfg["weaviate_url"])
    ensure_document_class(client)
    return client


def get_rag_chain(*, llm=None):
    """
    Build a Retrieval-QA chain. `llm` may be a streaming model.
    """
    embeddings = (
        OpenAIEmbeddings()
        if "OPENAI_API_KEY" in os.environ
        else HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    vectordb = Weaviate(_client(), "Document", "content", embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    return RetrievalQA.from_chain_type(
        llm=llm or get_llm(),
        retriever=retriever,
    )
