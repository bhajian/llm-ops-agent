from typing import Dict
from app.agents.rag_agent import run_rag_agent  # assumes this returns string


def rag_node(state: Dict) -> Dict:
    question = state.get("question", "")
    result = run_rag_agent(question)
    return {**state, "rag_result": result}
