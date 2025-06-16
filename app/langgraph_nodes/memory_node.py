from typing import Dict
from app.memory import retrieve_from_memory  # assumes this returns string


def memory_node(state: Dict) -> Dict:
    question = state.get("question", "")
    memory_result = retrieve_from_memory(question)
    return {**state, "memory_result": memory_result}
