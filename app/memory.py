import redis
import json
import logging

# Connect to Redis (adjust host/port/env if needed)
_r = redis.Redis(host="redis", port=6379, decode_responses=True)


def load_chat(chat_id: str) -> list:
    """
    Load chat history as a list of messages from Redis.
    """
    try:
        raw = _r.get(chat_id)
        return json.loads(raw) if raw else []
    except Exception as e:
        logging.error(f"Failed to load chat {chat_id}: {e}")
        return []


def save_chat(chat_id: str, messages: list) -> None:
    """
    Save chat history list of messages to Redis.
    """
    try:
        _r.set(chat_id, json.dumps(messages))
    except Exception as e:
        logging.error(f"Failed to save chat {chat_id}: {e}")


def retrieve_from_memory(question: str) -> str:
    """
    Basic keyword-based retrieval from stored conversations.
    For now, return a mock memory response. You can later extend this
    to use vector search, semantic matching, or memory indexing.
    """
    # This is just a placeholder – customize per your system
    return f"[Memory] I remember you asked something like: '{question}'"
