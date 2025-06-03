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
    # **CRITICAL CHANGE**: Increase 'k' to retrieve more relevant documents
    return vectordb.as_retriever(search_kwargs={"k": 8}) # Increased from 4 to 8

# ───────────────── public factory ────────────────────────────────────────
@lru_cache
def get_rag_chain(streaming: bool = False):
    """
    Builds and returns a modern, history-aware conversational RAG chain.
    """
    # **CRITICAL CHANGE**: Slightly higher temperature for better answer synthesis
    llm = get_llm(streaming=streaming, temperature=0.2) # Changed from 0.1 to 0.2

    retriever = get_weaviate_retriever()

    # 1. History-Aware Retriever Chain
    condense_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "Example: 'What is his role?' -> 'What is John Doe's role at Company X?'"
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
    # **CRITICAL CHANGE HERE: Strengthen the QA prompt for synthesis and strictness**
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Your goal is to answer the user's question **strictly and solely** based on the provided 'Context' documents. "
        "Synthesize information from multiple context snippets if necessary to form a complete and coherent answer. "
        "If the answer is not explicitly found within the provided context, state clearly and politely that you don't have enough information from the documents to answer. "
        "Do not use your general knowledge, make up information, infer details not explicitly present in the context, or say 'I don't know' without first confirming the information is absent from the provided context. "
        "Be concise, directly answer the question, and provide factual information only from the retrieved context. "
        "For questions about individuals (e.g., 'who is Behnam Hajian?', 'what does Matt Depue do?'), extract and present all relevant details about them from the context, such as their roles, experience, and affiliations. "
        "Do not apologize or act overly conversational; just provide the answer or state lack of information. "
        "\n\nContext:\n{context}"
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
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    return rag_chain

# ─── legacy alias for old code (optional, but good for compatibility) ────
get_cot_rag_chain = get_rag_chain