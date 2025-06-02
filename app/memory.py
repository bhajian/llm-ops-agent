# app/memory.py
import json, redis
from typing import List, Dict
from app.config import get_settings

_cfg = get_settings()
_r   = redis.Redis(host=_cfg["redis_host"], port=6379, decode_responses=True)

_KEY = "chat:{}"            # redis key template

def load_chat(cid: str) -> List[Dict]:
    raw = _r.get(_KEY.format(cid))
    return json.loads(raw) if raw else []

def save_chat(cid: str, user_msg: str, assistant_msg: str, scratch: str | None = None):
    """
    scratch = chain-of-thought text (stored but NEVER shown to user)
    """
    hist = load_chat(cid)
    hist.append({"role": "user",      "content": user_msg})
    hist.append({"role": "assistant", "content": assistant_msg,
                 "scratch": scratch})
    _r.set(_KEY.format(cid), json.dumps(hist), ex=60*60*24*7)
