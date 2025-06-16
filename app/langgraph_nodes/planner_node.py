from typing import Dict


def planner_node(state: Dict) -> Dict:
    question = state.get("question", "").lower()
    planner_result = []

    if any(keyword in question for keyword in ["now", "current time", "date", "today", "time in"]):
        planner_result.append("datetime")

    if any(keyword in question for keyword in ["remember", "memory", "yesterday", "before", "what did i say"]):
        planner_result.append("memory")

    if any(keyword in question for keyword in ["who", "what", "where", "why", "how", "search", "define"]):
        planner_result.append("rag")

    return {**state, "planner_result": planner_result}
