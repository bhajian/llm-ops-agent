# app/agents/rag_agent.py
from app.tools.vector_utils import search_vectorstore
from app.llm import call_llm

def run_rag_agent(question: str) -> str:
    docs = search_vectorstore(question)
    context = "\n".join(docs)
    return call_llm(prompt=f"Use this context to answer:\n{context}\n\nQ: {question}")
