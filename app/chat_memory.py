# app/chat_memory.py
# ────────────────────────────────────────────────────────────
"""
Minimal helper that feeds the last N turns plus the new question to whatever
LLM backend `get_llm()` returns. Compatible with vLLM / Llama / OpenAI.
"""

from typing import List

from langchain.schema import HumanMessage, AIMessage, SystemMessage # Import SystemMessage
from app.llm    import get_llm            # central factory (vllm, llama, etc.)
from app.memory import format_chat_history

WINDOW = 10  # how many back-turns to keep

# Define a system message for the general chat LLM
_CHAT_SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are a friendly, helpful, and concise conversational AI assistant. "
        "Engage in natural dialogue, answer questions directly, and avoid providing "
        "unnecessary technical explanations, code snippets, or lengthy examples. "
        "Keep your responses brief and to the point, focusing solely on the user's immediate query or conversational turn. "
        "Do not offer to explain concepts like LLMs, context windows, or learning types unless explicitly asked."
    )
)

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
    messages = [_CHAT_SYSTEM_MESSAGE] + _dicts_to_messages(history) # Add system message here
    messages.append(HumanMessage(content=query))

    resp = llm.invoke(messages)          # LC ≥0.1 works on list-of-messages
    # OpenAI / vLLM returns ChatMessage, HFTextGen returns str
    return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
