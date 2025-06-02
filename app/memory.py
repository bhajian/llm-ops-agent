# app/memory.py

import json
import redis
from typing import List, Dict, Any, Optional
from app.config import get_settings

_cfg = get_settings()

_r = redis.Redis(
    host=_cfg["redis_host"],
    port=_cfg["redis_port"],
    db=_cfg["redis_db"],
    decode_responses=True
)

_KEY = "chat:{}"  # Redis key pattern

def load_chat(cid: str) -> List[Dict[str, Any]]:
    raw = _r.get(_KEY.format(cid))
    return json.loads(raw) if raw else []

def save_chat(cid: str, user_msg: str, assistant_msg: str, scratch: Optional[str] = None):
    """
    Appends user/assistant messages to chat history and persists to Redis.
    `scratch` stores CoT/logits/internal info (never shown to user).
    """
    hist = load_chat(cid)
    hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": assistant_msg, "scratch": scratch})
    _r.set(_KEY.format(cid), json.dumps(hist), ex=_cfg["redis_ttl"])

def format_chat_history(history: List[Dict[str, Any]], limit: int = 10) -> str:
    """
    Returns a printable string from recent messages.
    Omits scratch/debug fields.
    """
    return "\n".join(
        f"{m['role'].title()}: {m['content']}"
        for m in history[-limit:]
        if m.get("content")
    )
