# app/agents/reasoner_agent.py
from app.llm import call_llm

def run_reasoner_agent(state: dict) -> str:
    chunks = []
    if "rag_result" in state:
        chunks.append("Knowledge:\n" + state["rag_result"])
    if "memory_result" in state:
        chunks.append("Memory:\n" + state["memory_result"])
    if "datetime_result" in state:
        chunks.append("DateTime:\n" + state["datetime_result"])

    combined = "\n\n".join(chunks)
    return call_llm(prompt=f"Answer based on:\n{combined}")
