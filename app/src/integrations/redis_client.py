"""
redis_client.py
---------------
Minimal Redis helper + “Blackboard” wrapper.

Env:
    REDIS_URL   (default: redis://localhost:6379/0)
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict

import redis

_log = logging.getLogger("Blackboard")

# ── connection helper ──────────────────────────────
_singleton: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _singleton
    if _singleton is None:
        _singleton = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
    return _singleton


# ── Blackboard wrapper ─────────────────────────────
class Blackboard:
    """
    {conversation_id}:{slot}  →  Redis HASH
    """

    def __init__(
        self,
        r: redis.Redis,
        *,
        cid_prefix: str = "",
        ttl: int = 0,
    ):
        self.redis = r
        self.cid_prefix = cid_prefix
        self.ttl = ttl

    # public ----------------------------------------------------------
    def write(self, cid: str | None, slot: str, mapping: Dict[str, Any]) -> None:
        key = self._key(cid, slot)
        _log.debug("WRITE %s %s → %s", cid or self.cid_prefix, slot, mapping)
        with self.redis.pipeline() as p:
            for k, v in mapping.items():
                p.hset(key, k, json.dumps(v))
            if self.ttl:
                p.expire(key, self.ttl)
            p.execute()

    def read_all(self, cid: str | None = None) -> Dict[str, Dict[str, Any]]:
        prefix = self._cid(cid)
        out: Dict[str, Dict[str, Any]] = {}
        for key in self.redis.keys(f"{prefix}*"):
            slot = key[len(prefix):]
            raw = self.redis.hgetall(key)
            out[slot] = {k: json.loads(v) for k, v in raw.items()}
        _log.debug("READ  %s → %s", cid or self.cid_prefix, out)
        return out

    def bind(self, cid: str) -> "Blackboard":
        """Return a view where *cid* is implicit."""
        return Blackboard(self.redis, cid_prefix=f"{cid}:", ttl=self.ttl)

    # helpers ---------------------------------------------------------
    def _cid(self, cid: str | None) -> str:
        return f"{cid}:" if cid else self.cid_prefix

    def _key(self, cid: str | None, slot: str) -> str:
        return f"{self._cid(cid)}{slot}"
