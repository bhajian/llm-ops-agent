# app/agents/planner_agent.py
from __future__ import annotations
import json
from typing import List, Dict
from app.llm import chat_llm               # already available

SYS = """You are PlannerGPT, an expert at breaking a complex
user request into minimal, ordered steps.  ALWAYS respond with
pure JSON:   { "steps": [ "<task 1>", "<task 2>", … ] }"""

async def make_plan(question: str, history: List[Dict[str, str]]) -> List[str]:
    """Return a list of tool-free textual subtasks."""
    conv = [{"role": "system", "content": SYS},
            *history[-4:],                             # recent context helps
            {"role": "user", "content": question}]
    resp = await chat_llm(conv)
    try:
        return json.loads(resp)["steps"]
    except Exception:
        # Fallback – single-step plan
        return [question]
