"""
llm_agent.py
============
Generates the assistant’s reply.

• Builds a prompt that always includes facts from MemoryAgent.
• Supports blocking or streaming (async SSE).
• Whitespace-safe: new-lines → spaces, never starts an SSE line with a space,
  inserts one spacer when needed so words never glue together.
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import AsyncIterator, List, Union

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessageChunk,
)

from ..llm import get_llm
from .memory_agent import MemoryAgent

_log = logging.getLogger("LlmAgent")

# ────────────────── configuration ──────────────────
_SYS_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful AI assistant.")

_LLM_BLOCK  = get_llm(streaming=False, temperature=0.7)
_LLM_STREAM = get_llm(streaming=True,  temperature=0.7)


class LlmAgent:                                        # noqa: D101
    """Central component that converts a user message into an assistant reply."""

    def __init__(self, memory: MemoryAgent):
        self.memory = memory

    # ───────── prompt builder ─────────
    def _build_chat(self, cid: str, user_msg: str) -> List:
        ctx = self.memory.context(cid)
        sys_parts = [_SYS_PROMPT]
        if name := ctx.get("user_name"):
            sys_parts.append(f"The user's name is {name}.")
        return [
            SystemMessage(content=" ".join(sys_parts)),
            HumanMessage(content=user_msg),
        ]

    # ───────── public API ─────────
    def run(
        self,
        cid: str,
        user_msg: str,
        *,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        chat_in = self._build_chat(cid, user_msg)
        _log.debug("Prompt for %s → %s", cid, chat_in)

        # ---------- streaming ----------
        if stream:

            async def _aiter() -> AsyncIterator[str]:
                last_sent_char = ""  # last char already yielded to client

                async for chunk in _LLM_STREAM.astream(chat_in):
                    if not isinstance(chunk, AIMessageChunk):
                        continue
                    part: str = chunk.content or ""
                    if not part:
                        continue

                    # 1) Replace any newline with single space (so they're visible).
                    part = part.replace("\n", " ")

                    # 2) Remove *leading* spaces – browsers drop them after "data:".
                    while part.startswith(" "):
                        if last_sent_char and last_sent_char != " ":
                            # need exactly one spacer
                            yield " "
                            last_sent_char = " "
                        part = part[1:]

                    if not part:
                        continue

                    # 3) Ensure word boundary across chunks.
                    if (
                        last_sent_char
                        and last_sent_char.isalnum()
                        and part[0].isalnum()
                    ):
                        yield " "
                        last_sent_char = " "

                    # 4) Yield the cleaned chunk.
                    yield part
                    last_sent_char = part[-1]
                    await asyncio.sleep(0)  # flush

            return _aiter()

        # ---------- blocking ----------
        return _LLM_BLOCK.invoke(chat_in).content
