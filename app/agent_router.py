# app/agent_router.py
"""
Orchestrates the agentic flow (planner → RAG → answer) and gracefully
handles legacy chat-history records that may store the text under "msg".
"""
from __future__ import annotations
import asyncio, logging
from typing import List, Dict

from app.agents.planner_agent import make_plan
from app.agents.rag_agent import _RETRIEVER, _format_docs
from app.memory import load_chat
from app.llm import chat_llm


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _msg_text(m: Dict[str, str]) -> str:
    """Return the message text regardless of whether it is saved as
    `content` (new) or `msg` (old)."""
    return m.get("content") or m.get("msg") or ""


async def _answer_step(step: str, history: List[Dict[str, str]]) -> str:
    """Embed step → nearVector RAG search → answer (or fallback)."""
    try:
        docs = await _RETRIEVER.ainvoke(step)
    except Exception as exc:
        logging.warning("RAG retrieval failed: %s", exc)
        docs = []

    msgs: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a concise, knowledgeable assistant. "
                "Answer the user question helpfully and accurately."
            ),
        }
    ]

    if docs:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "Background context from the knowledge base "
                    "(use if relevant):\n\n" + _format_docs(docs)
                ),
            }
        )

    # most-recent 6 turns for context
    for m in history[-6:]:
        msgs.append({"role": m.get("role", "user"), "content": _msg_text(m)})

    msgs.append({"role": "user", "content": step})
    return await chat_llm(msgs)


# --------------------------------------------------------------------------- #
# Public entry-point                                                          #
# --------------------------------------------------------------------------- #
async def run_agentic_chat(query: str, chat_id: str) -> str:
    """Main orchestrator called from app.main."""
    history = load_chat(chat_id)

    try:
        plan: List[str] = make_plan(query, history, chat_id) or [query]
    except Exception as exc:
        logging.warning("Planner failed (%s). Falling back to direct answer.", exc)
        plan = [query]

    answers: List[str] = []
    for step in plan:
        answers.append(
            await _answer_step(step, history + [{"role": "user", "content": query}])
        )

    return answers[-1] if answers else "Sorry, I couldn't find an answer."
