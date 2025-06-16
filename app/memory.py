# app/memory.py
"""
Unified memory helper (sync + async)
────────────────────────────────────
• New preferred call:   await save_chat(cid, role="user", msg="Hi")
• Old pair style still works: await save_chat(cid, "Hi", "Hello")
• load_recent / load_chat unchanged
"""
from __future__ import annotations
import json, os, inspect
from typing import List, Dict

import redis
import redis.asyncio as aredis

REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
_r_sync     = redis.from_url(REDIS_URL, decode_responses=True)
_r_async    = aredis.from_url(REDIS_URL, decode_responses=True)
CHAT_KEY    = lambda cid: f"chat:{cid}"


# ─────────────────────────────── save helpers ──────────────────────────────
async def _push(cid: str, role: str, msg: str) -> None:
    """Internal: append a single turn."""
    await _r_async.rpush(CHAT_KEY(cid), json.dumps({"role": role, "msg": msg}))


async def save_chat(cid: str, *args, **kw) -> None:
    """
    Flexible saver:

    • NEW  → await save_chat(cid, role="user", msg="Hi")
    • OLD  → await save_chat(cid, "Hi", "Hello")
             (user-msg first, assistant-msg second)
    """
    # --- new keyword style -------------------------------------------------
    if "role" in kw and "msg" in kw:
        await _push(cid, kw["role"], kw["msg"])
        return

    # --- legacy two-arg style ---------------------------------------------
    if len(args) == 2:
        user, assistant = args          # type: ignore
        await _push(cid, "user", user)
        await _push(cid, "assistant", assistant)
        return

    raise TypeError(
        "save_chat() expects (cid, role=…, msg=…)  or  (cid, user, assistant)"
    )


# ───────────────────────── recent / full history ──────────────────────────
async def load_recent(cid: str, k: int = 12) -> List[Dict[str, str]]:
    raw = await _r_async.lrange(CHAT_KEY(cid), -k, -1)
    return [json.loads(x) for x in raw]


def load_chat(cid: str) -> List[Dict[str, str]]:
    raw = _r_sync.lrange(CHAT_KEY(cid), 0, -1)
    return [json.loads(x) for x in raw]


# keep old alias used by /history routes
_r = _r_sync
