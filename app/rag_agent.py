# app/rag_agent.py
"""
Backend-agnostic, CONVERSATIONAL RAG module.
This version uses a modern, history-aware retrieval chain to provide
contextually-aware answers based on chat history and retrieved documents.
"""

from __future__ import annotations
from functools import lru_cache

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_weaviate.vectorstores import WeaviateVectorStore

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
    """Singleton embedding model."""
    return get_embeddings()

def get_weaviate_retriever():
    """Creates and returns a Weaviate vector store retriever."""
    vectordb = WeaviateVectorStore(
        client=_client(),
        index_name="Document",
        text_key="text",
        embedding=_embedding(),
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})


# ───────────────── public factory ────────────────────────────────────────
@lru_cache
def get_rag_chain(streaming: bool = False):
    """
    Builds and returns a modern, history-aware conversational RAG chain.
    """
    llm = get_llm(streaming=streaming, temperature=0.1) # Lower temp for factual RAG
    retriever = get_weaviate_retriever()

    # 1. History-Aware Retriever Chain
    # This chain takes the conversation history and a new question, and then
    # uses an LLM to create a standalone search query.
    condense_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_q_prompt
    )

    # 2. Question Answering Chain
    # This chain takes the original question, retrieved documents, and history,
    # and generates a final answer.
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Be concise and ground your answer in the provided context.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Full Conversational RAG Chain
    # This final chain combines the two previous chains. When invoked, it will:
    #   a. Run the history_aware_retriever to get a new query and retrieve documents.
    #   b. Pass the retrieved documents and original inputs to the Youtube_chain to get an answer.
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    return rag_chain

# ─── legacy alias for old code (optional, but good for compatibility) ────
get_cot_rag_chain = get_rag_chain
