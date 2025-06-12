# app/agents/rag_agent.py
"""
Backend-agnostic, CONVERSATIONAL RAG module.
Exposes functions for LangGraph nodes: condense question, retrieve documents, answer from context.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Any

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_weaviate.vectorstores import WeaviateVectorStore

# Local imports (ADJUSTED PATHS)
from app.config import get_settings
from app.llm import get_llm, get_embeddings
from app.tools.weaviate_v4 import connect_weaviate, ensure_document_class # Tool remains in tools/

import asyncio # Needed for asyncio.to_thread


_cfg = get_settings()

# LLM for internal RAG steps (e.g., condensing question) - usually non-streaming
_rag_internal_llm = get_llm(streaming=False, temperature=0.2)

# LLM for RAG final answer synthesis - MUST be streaming
_rag_synthesis_llm = get_llm(streaming=True, temperature=0.2)


# ───────────────── Helpers for Weaviate and Embeddings ─────────────────
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
    return vectordb.as_retriever(search_kwargs={"k": 8}) # Increased from 4 to 8


# ───────────────── RAG Node Functions ──────────────────────────────────
async def condense_question_for_rag(user_query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Condenses chat history and the latest user question into a standalone question for RAG retrieval.
    """
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
    
    # Convert dict history to LangChain messages
    lc_chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in chat_history
    ]

    condensed_question_messages = condense_q_prompt.format_messages(
        input=user_query,
        chat_history=lc_chat_history
    )
    
    # Use the internal LLM (non-streaming for this step)
    condensed_question_llm_response = await asyncio.to_thread(_rag_internal_llm.invoke, condensed_question_messages)
    return condensed_question_llm_response.content.strip() if hasattr(condensed_question_llm_response, 'content') else str(condensed_question_llm_response).strip()


async def answer_question_from_context(original_query: str, context: str) -> str:
    """
    Synthesizes the final answer based on the original query and retrieved context.
    This function's LLM will be streaming.
    """
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Your goal is to answer the user's question **strictly and solely** based on the provided 'Context' documents. "
        "Synthesize information from multiple context snippets if necessary to form a complete and coherent answer. "
        "If the answer is found within the context, explicitly state that the information is from the documents. "
        "If the answer is not explicitly found within the provided context, state clearly and politely that you don't have enough information from the documents to answer. "
        "Do not use your general knowledge, make up information, infer details not explicitly present in the context, or say 'I don't know' without first confirming the information is absent from the provided context. "
        "Be concise, directly answer the question, and provide factual information only from the retrieved context. "
        "For questions about individuals (e.g., 'who is Behnam Hajian?', 'what does Matt Depue do?'), extract and present all relevant details about them from the context, such as their roles, experience, and affiliations. "
        "Do not apologize or act overly conversational; just provide the answer or state lack of information. "
        "DO NOT include ANY conversational text, preambles, questions, or explanations beyond the direct answer. Do not over-generate."
        "\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"), # Input here is the original user query for context
        ]
    )
    final_qa_messages = qa_prompt.format_messages(
        input=original_query,
        context=context
    )

    # Use the streaming LLM for the final answer synthesis
    final_answer_llm_response = await asyncio.to_thread(_rag_synthesis_llm.invoke, final_qa_messages)
    return final_answer_llm_response.content.strip() if hasattr(final_answer_llm_response, 'content') else str(final_answer_llm_response).strip()

