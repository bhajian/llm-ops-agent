# --- app/memory.py ---
import redis
import json
import os

r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)

def save_chat(chat_id, query, response):
    r.rpush(chat_id, json.dumps({"query": query, "response": response}))

def load_chat(chat_id):
    history = r.lrange(chat_id, 0, -1)
    return [json.loads(item) for item in history] if history else []
