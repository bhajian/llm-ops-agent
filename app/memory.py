"""
app/memory.py
──────────────────────────────────────────────────────────────
Unified memory layer.

• save_chat   – async  (used by the agent core)
• load_recent – async  (used by reasoner for context)
• load_chat   – sync   (needed by old /history & greeting logic)

Both helpers talk to the same Redis instance but with different
clients so we don’t block the event loop.
"""
from __future__ import annotations
import json, os
from typing import List, Dict

import redis                          # blocking client
import redis.asyncio as aredis        # asyncio client

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# two connections, same DB
_r_sync = redis.from_url(REDIS_URL, decode_responses=True)
_r_async = aredis.from_url(REDIS_URL, decode_responses=True)

CHAT_KEY = lambda cid: f"chat:{cid}"


# ─────────────────────────────────────────────────────────────
# async helpers (agentic core)
# ─────────────────────────────────────────────────────────────
async def save_chat(cid: str, user: str, assistant: str) -> None:
    await _r_async.rpush(
        CHAT_KEY(cid),
        json.dumps({"role": "user", "msg": user}),
        json.dumps({"role": "assistant", "msg": assistant}),
    )


async def load_recent(cid: str, k: int = 12) -> List[Dict[str, str]]:
    raw = await _r_async.lrange(CHAT_KEY(cid), -2 * k, -1)
    return [json.loads(x) for x in raw]


# ─────────────────────────────────────────────────────────────
# sync helper for backwards-compat routes/UI
# ─────────────────────────────────────────────────────────────
def load_chat(cid: str) -> List[Dict[str, str]]:
    raw = _r_sync.lrange(CHAT_KEY(cid), 0, -1)
    return [json.loads(x) for x in raw]


# Expose the blocking client under the old name so
# /history/list continues to work unchanged.
_r = _r_sync
