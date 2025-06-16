from typing import Dict
from app.agents.reasoner_agent import run_reasoner_agent  # assumes this takes full state and returns final str


def reasoner_node(state: Dict) -> Dict:
    final = run_reasoner_agent(state)
    return {**state, "final_answer": final}
