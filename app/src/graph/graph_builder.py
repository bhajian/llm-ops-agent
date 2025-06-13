"""
graph_builder.py – Dispatcher for each chat turn.
"""
from __future__ import annotations
import logging
from typing import AsyncIterator, Union

from ..agents.memory_agent import MemoryAgent
from ..agents.llm_agent import LlmAgent

_log = logging.getLogger("Graph")

_mem = MemoryAgent(llm=None)
_llm = LlmAgent(memory=_mem)


def handle_chat(
    message: str,
    conversation_id: str,
    *,
    stream: bool = False,
) -> Union[str, AsyncIterator[str]]:
    _log.info("▶️  %s stream=%s msg=%s", conversation_id, stream, message[:80])
    _mem.update(conversation_id, message)
    reply = _llm.run(conversation_id, message, stream=stream)
    _log.info("◀️  %s replied %s", conversation_id, "stream" if stream else "block")
    return reply
