# app/rag_agent.py
import os
import weaviate

from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain.chains import RetrievalQA

from app.llm import get_llm
from app.config import get_settings

def get_rag_chain(llm=None):
    """
    Build a Retrieval-QA chain. If `llm` is provided (e.g., streaming version)
    it will be used; otherwise we create the default non-stream model.
    """
    cfg = get_settings()
    client = weaviate.Client(cfg["weaviate_url"])

    embeddings = (
        OpenAIEmbeddings()
        if "OPENAI_API_KEY" in os.environ
        else HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    vectordb  = Weaviate(client, "Document", "content", embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    return RetrievalQA.from_chain_type(
        llm=llm or get_llm(),
        retriever=retriever
    )
