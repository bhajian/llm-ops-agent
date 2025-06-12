"""Redis connection factory + Blackboard helper."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Dict

import redis

from ..config import get_settings

_cfg = get_settings()


def get_redis() -> redis.Redis:
    return redis.from_url(_cfg.redis_url, decode_responses=True)


class Blackboard:
    """
    Ephemeral shared store for agent-to-agent hand-off.
    Keys auto-expire (TTL) so stale data never pollutes the next turn.
    """

    def __init__(self, conn: redis.Redis, ttl: int = 60):
        self.r = conn
        self.ttl = ttl  # seconds

    # -- high-level helpers --
    def write(self, convo_id: str, slot: str, payload: dict[str, Any]):
        key = f"bb:{convo_id}:{slot}"
        self.r.setex(key, timedelta(seconds=self.ttl), json.dumps(payload))

    def read_all(self, convo_id: str) -> Dict[str, Any]:
        pattern = f"bb:{convo_id}:*"
        out: Dict[str, Any] = {}
        for key in self.r.scan_iter(pattern):
            slot = key.split(":")[-1]
            out[slot] = json.loads(self.r.get(key) or "{}")
        return out

    def clear(self, convo_id: str):
        pattern = f"bb:{convo_id}:*"
        for key in self.r.scan_iter(pattern):
            self.r.delete(key)
