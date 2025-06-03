# app/chat_memory.py
# ────────────────────────────────────────────────────────────
"""
Minimal helper that feeds the last N turns plus the new question to whatever
LLM backend `get_llm()` returns. Compatible with vLLM / Llama / OpenAI.
"""

from typing import List

from langchain.schema import HumanMessage, AIMessage
from app.llm    import get_llm            # central factory (vllm, llama, etc.)
from app.memory import format_chat_history

WINDOW = 10  # how many back-turns to keep


def _dicts_to_messages(history: List[dict]):
    msgs = []
    for m in history[-WINDOW:]:
        role, content = m.get("role"), m.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def chat_with_memory(query: str, history: List[dict]) -> str:
    llm = get_llm()                       # any backend
    messages = _dicts_to_messages(history)
    messages.append(HumanMessage(content=query))

    resp = llm.invoke(messages)          # LC ≥0.1 works on list-of-messages
    # OpenAI / vLLM returns ChatMessage, HFTextGen returns str
    return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
