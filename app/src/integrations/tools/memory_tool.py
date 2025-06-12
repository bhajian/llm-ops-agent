"""
MemorySummaryTool
─────────────────
Returns the rolling 250-token summary stored in Redis for a given
conversation_id.  Uses Pydantic-v2 PrivateAttr to store the Redis handle.
"""

from typing import Any, ClassVar

from langchain.tools import BaseTool
from pydantic import PrivateAttr

from ...config import get_settings
from ..redis_client import get_redis

_cfg = get_settings()
_redis = get_redis()


class MemorySummaryTool(BaseTool):
    # LangChain / BaseTool metadata  (must be ClassVar with Pydantic-v2)
    name: ClassVar[str] = "memory_summary"
    description: ClassVar[str] = (
        "Return the 250-token conversation summary. "
        "Args: conversation_id(str)."
    )

    # Private attribute (ignored by Pydantic field validation)
    _r: Any = PrivateAttr()

    # ─────────────────── constructor ────────────────────
    def __init__(self, redis_conn=None):
        super().__init__()
        self._r = redis_conn or _redis

    # ─────────────────── BaseTool sync hook ─────────────
    def _run(self, conversation_id: str, **_: Any) -> str:
        return self._r.get(f"convsum:{conversation_id}") or ""

    # ─────────────────── BaseTool async hook ────────────
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported for MemorySummaryTool.")
