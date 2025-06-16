# app/agents/reasoner_agent.py
from __future__ import annotations
from typing import List, Dict
from app.llm import chat_llm

STEP_SYS = "You are a domain expert.  Use the supplied context to answer."
FINAL_SYS = """You are a senior analyst.  Given the user question,
the step answers, and the chat history, deliver a concise final reply."""

# ––––– per-step –––––
async def solve_step(step: str,
                     ctx: str,
                     history: List[Dict[str, str]]) -> str:
    prompt = [{"role": "system", "content": STEP_SYS},
              {"role": "system", "content": f"Context:\n{ctx}"},
              {"role": "user",   "content": step}]
    return await chat_llm(prompt)

# ––––– aggregation –––––
async def compose_final(question: str,
                        step_answers: List[str],
                        history: List[Dict[str, str]]) -> str:
    joined = "\n\n".join(f"{i+1}. {a}" for i, a in enumerate(step_answers))
    prompt = [{"role": "system", "content": FINAL_SYS},
              *history[-4:],
              {"role": "system", "content": f"Step answers:\n{joined}"},
              {"role": "user",   "content": question}]
    return await chat_llm(prompt)
