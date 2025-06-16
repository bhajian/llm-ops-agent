# app/agent_router.py
"""
Orchestrates the agentic flow (planner → RAG → answer).

Changes
-------
● Added `_maybe_quick_answer()` that asks a lightweight LLM “routing prompt”
  to decide if the user just wants the current time/date.
● If the LLM replies `YES`, we immediately return the formatted timestamp.
  Otherwise we continue with the normal planner → RAG pipeline.
● No regexp or keyword lists are hard-coded – the LLM does the intent
  detection, so indirect queries like *“Could you tell me what day it is?”*
  also work.

You can extend `_QUICK_TASKS` with additional intents/answers later without
touching the core logic.
"""
from __future__ import annotations
import logging, asyncio
from datetime import datetime
from typing import List, Dict, Optional

from app.agents.planner_agent import make_plan
from app.agents.rag_agent    import _RETRIEVER, _format_docs
from app.memory              import load_chat
from app.llm                 import chat_llm


# ────────────────────────── util helpers ─────────────────────────
def _msg_text(m: Dict[str, str]) -> str:
    """Return message text regardless of old/new field name."""
    return m.get("content") or m.get("msg") or ""


# ─────────────────── quick-answer intent detector ───────────────
_QUICK_TASKS = {
    "time_or_date": {
        "question": (
            "Does the user's last message ask for the *current* time or date? "
            "Answer ONLY with 'YES' or 'NO'."
        ),
        "answer_fn": lambda: datetime.now().astimezone().strftime(
            "It is %H:%M:%S on %A, %B %d, %Y (%Z)."
        ),
    },
    # You can add more lightweight intents here later
}


async def _maybe_quick_answer(query: str) -> Optional[str]:
    """
    Ask a tiny routing-prompt to decide if we can answer locally.
    Returns the quick answer string, or *None* if we should continue with
    the full agentic flow.
    """
    for task in _QUICK_TASKS.values():
        routing_prompt = [
            {"role": "system", "content": task["question"]},
            {"role": "user",   "content": query},
        ]
        try:
            verdict = (await chat_llm(routing_prompt)).strip().upper()
        except Exception as err:
            logging.warning("Routing LLM failed (%s) – skipping quick path", err)
            continue

        if verdict.startswith("YES"):
            return task["answer_fn"]()
    return None


# ─────────────────── per-step answer helper ─────────────────────
async def _answer_step(step: str, history: List[Dict[str, str]]) -> str:
    """RAG retrieval + final reasoning for a single step."""
    try:
        docs = await _RETRIEVER.ainvoke(step)
    except Exception as exc:
        logging.warning("RAG retrieval failed: %s", exc)
        docs = []

    msgs: List[Dict[str, str]] = [
        {"role": "system",
         "content": (
             "You are a concise, knowledgeable assistant. "
             "Answer the user helpfully and accurately."
         )},
    ]

    if docs:
        msgs.append(
            {"role": "system",
             "content": "Background context (use if relevant):\n\n" + _format_docs(docs)}
        )

    # last 6 turns for conversational context
    for m in history[-6:]:
        msgs.append({"role": m.get("role", "user"), "content": _msg_text(m)})

    msgs.append({"role": "user", "content": step})
    return await chat_llm(msgs)


# ───────────────────── main public entry-point ──────────────────
async def run_agentic_chat(query: str, chat_id: str) -> str:
    """
    Handles one user turn.

    1. Try cheap quick-answers (time/date, etc.)
    2. Else:   planner → RAG → LLM reasoning
    """
    # 1) quick-answer fast path
    quick = await _maybe_quick_answer(query)
    if quick is not None:
        return quick

    # ------------------------------------------------------------------
    # 2) autonomous multi-step reasoning
    # ------------------------------------------------------------------
    history = load_chat(chat_id)

    try:
        plan: List[str] = make_plan(query, history, chat_id) or [query]
    except Exception as exc:
        logging.warning("Planner failed (%s). Falling back to direct answer.", exc)
        plan = [query]

    answers: List[str] = []
    for step in plan:
        if step.strip():
            ans = await _answer_step(step, history + [{"role": "user",
                                                       "content": query}])
            answers.append(ans)
            # stream assistant turn into pseudo-history
            history.append({"role": "assistant", "content": ans})

    return answers[-1] if answers else "Sorry, I couldn't find an answer."
