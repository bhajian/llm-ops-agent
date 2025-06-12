"""
graph_builder.py
────────────────
PlannerAgent → task agents → ReasonerAgent pipeline.

* Handles malformed planner JSON.
* Converts any iterator/generator from Reasoner into a string.
* Provides a friendly fallback answer if Reasoner returns empty text.
"""

from __future__ import annotations

from typing import Iterator, List

from langchain.schema import AIMessage

from ..llm import get_llm
from ..agents import (
    planner_agent,      # callable
    retrieval_agent,
    finance_agent,
    k8s_agent,
    reasoner_agent,
)
from ..integrations import blackboard, redis_conn

# ─────────────────── helper funcs ───────────────────
def _update_summary(convo_id: str, user_q: str, answer: str) -> None:
    prev = redis_conn.get(f"convsum:{convo_id}") or ""
    llm = get_llm(streaming=False, temperature=0.0, max_tokens=256)

    prompt = (
        "Summarise the conversation so far in ≤250 tokens.\n"
        f"Existing summary:\n{prev}\n\n"
        f"New exchange:\nUser: {user_q}\nAssistant: {answer}\n"
    )
    redis_conn.set(f"convsum:{convo_id}", llm.invoke(prompt).content)


def _run_task(task_id: str, user_q: str, convo_id: str, stream: bool):
    fn = {
        "retrieval": retrieval_agent,
        "finance": finance_agent,
        "k8s": k8s_agent,
    }.get(task_id)
    if fn is None:
        return []

    return (
        fn(user_q, convo_id, stream=True)
        if stream
        else (fn(user_q, convo_id, stream=False) or [])
    )

# ─────────────────── orchestrator ───────────────────
def handle_chat(
    user_query: str,
    conversation_id: str,
    *,
    stream: bool = False,
) -> str | Iterator[str]:
    # 1 — Planner
    tasks = planner_agent(user_query, conversation_id)

    # 2 — Execute tasks
    if stream:
        for tid in tasks:
            for chunk in _run_task(tid, user_query, conversation_id, True):
                if isinstance(chunk, AIMessage):
                    yield chunk.content
        if tasks:
            yield "\n---\n"
    else:
        for tid in tasks:
            _run_task(tid, user_query, conversation_id, False)

    # 3 — Reasoner
    if stream:
        final_chunks: List[str] = []
        for chunk in reasoner_agent(user_query, conversation_id, stream=True):
            if isinstance(chunk, AIMessage):
                final_chunks.append(chunk.content)
                yield chunk.content
        final_answer = "".join(final_chunks)
    else:
        final_answer = reasoner_agent(user_query, conversation_id, stream=False)

        # ensure final_answer is a string
        if not isinstance(final_answer, str):
            chunks = [
                ck.content if hasattr(ck, "content") else str(ck)
                for ck in final_answer
            ]
            final_answer = "".join(chunks)

    # 4 — Fallback for empty answer
    if not final_answer.strip():
        final_answer = "Hello! How can I help you today?"

    # 5 — Update memory & cleanup
    _update_summary(conversation_id, user_query, final_answer)
    blackboard.clear(conversation_id)

    if not stream:
        return final_answer
