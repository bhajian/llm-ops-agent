"""
PlannerAgent
============
Generates an ordered list of task IDs in JSON.
If the LLM returns malformed JSON, we gracefully return an empty list,
allowing graph_builder to choose a safe default behaviour.
"""

import json
import re
from typing import List

from ..llm import get_llm
from ..integrations.tools import MemorySummaryTool

_TASK_IDS = {"retrieval", "finance", "k8s"}

# ────────────────── helpers ──────────────────
def _safe_parse_tasks(text: str) -> List[str]:
    """
    Extract first {...} JSON block and return validated task list.
    On any failure, return []  (caller will decide fallback).
    """
    try:
        block = re.search(r"\{.*?\}", text, re.S)
        if block is None:
            raise ValueError("no JSON found")

        data = json.loads(block.group(0))
        tasks = [
            t["id"] if isinstance(t, dict) else t
            for t in data.get("tasks", [])
        ]
        return [tid for tid in tasks if tid in _TASK_IDS]
    except Exception as err:          # noqa: BLE001
        print(f"⚠️  Planner JSON parse failed: {err}")
        return []


# ────────────────── main entry ──────────────────
def run(user_query: str, conversation_id: str) -> List[str]:
    summary_tool = MemorySummaryTool()
    summary = summary_tool.run(conversation_id)

    llm = get_llm(streaming=False, temperature=0.0)
    prompt = (
        "You are PlannerAgent. Choose which domain agents to run.\n"
        "Allowed ids: retrieval, finance, k8s.\n"
        'Return JSON e.g. {"tasks":[{"id":"finance"}]}.\n\n'
        f"Conversation summary:\n{summary}\n\n"
        f"User question:\n{user_query}\n"
    )

    resp = llm.invoke(prompt)
    return _safe_parse_tasks(resp.content)
