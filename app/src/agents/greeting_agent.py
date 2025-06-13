"""
GreetingAgent
=============
Fast regex answers + LLM fallback (via llm.py).

Environment variables
---------------------
LLM_SYSTEM_PROMPT   – system prompt for the model (optional)
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Union

from langchain_core.messages import SystemMessage, HumanMessage  # LangChain MSGs
from ..llm import get_llm
from .base_agent import BaseAgent
from .memory_agent import _MEMORY

# ── config ─────────────────────────────────────────────────────────────
_SYS_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a helpful assistant. Respond concisely in English.",
)

# Instantiate chat models once
_LLM_BLOCK = get_llm(streaming=False, temperature=0.7)
_LLM_STREAM = get_llm(streaming=True, temperature=0.7)


class GreetingAgent(BaseAgent):                # noqa: D101
    name = "greeting"

    # pre-compiled patterns
    _HELLO     = re.compile(r"\b(hi|hello|hey)\b", re.I)
    _HOW_ARE_U = re.compile(r"\bhow (?:are|r) (?:you|u)\b", re.I)
    _ASK_LAST  = re.compile(r"\bwhat did (?:i|you) (?:just )?ask\b", re.I)

    # ------------------------------------------------------------------
    def run(                                   # noqa: D401
        self,
        cid: str,
        message: str,
        *,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        • Returns str when stream=False  (/chat)
        • Returns iterator when stream=True  (/chat/stream)
        """
        store = _MEMORY[cid]
        name = store.get("user_name")

        # 1) lightning-fast regex replies
        if self._HELLO.search(message):
            return f"Hello {name}!" if name else "Hello! How can I help you?"

        if self._HOW_ARE_U.search(message):
            return "I’m just a bunch of code, but thanks for asking—how are you?"

        if self._ASK_LAST.search(message):
            prev = store.get("last_user")
            return f"You asked: “{prev}”" if prev else "I don’t recall your last question."

        # 2) fallback → LLM (LangChain chat model)
        chat_input = [
            SystemMessage(content=_SYS_PROMPT),
            HumanMessage(content=message),
        ]

        if stream:
            def _gen() -> Iterator[str]:
                for chunk in _LLM_STREAM.stream(chat_input):
                    if chunk.content:          # AIMessageChunk
                        yield chunk.content
            return _gen()

        # blocking path
        response = _LLM_BLOCK.invoke(chat_input)   # AIMessage
        return response.content
