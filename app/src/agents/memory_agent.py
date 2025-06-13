"""
MemoryAgent – LLM-extracted facts persisted in Redis.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from ..llm import get_llm
from ..integrations.redis_client import get_redis, Blackboard
from .base_agent import BaseAgent

_log = logging.getLogger("MemoryAgent")

_EXTRACT = get_llm(streaming=False, temperature=0)
_SYS = (
    "Extract user facts. Return ONLY JSON. "
    'Keys: {"user_name": str|null}. If none present return {}.'
)
_BB = Blackboard(get_redis(), ttl=0)        # keep forever
_SLOT = "facts"


class MemoryAgent(BaseAgent):                          # noqa: D101
    name = "memory"

    def _bb(self, cid: str) -> Blackboard:
        return _BB.bind(cid)

    # --------------------------------------------------------------
    def update(self, cid: str, user_msg: str) -> None:
        _log.debug("⬅️  %s user: %s", cid, user_msg)
        chat = [SystemMessage(content=_SYS), HumanMessage(content=user_msg)]
        try:
            data: Dict[str, Any] = json.loads(_EXTRACT.invoke(chat).content)
        except json.JSONDecodeError:
            _log.warning("Extractor returned non-JSON")
            return
        _log.debug("LLM extract → %s", data)
        if data:
            self._bb(cid).write(None, _SLOT, data)

    def context(self, cid: str) -> Dict[str, Any]:
        ctx = self._bb(cid).read_all().get(_SLOT, {})
        _log.debug("Context for %s → %s", cid, ctx)
        return ctx
