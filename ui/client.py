import os
import requests

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")
AUTH_HEADER = {"Authorization": os.getenv("AUTH_TOKEN", "supersecrettoken")}


def list_chats():
    try:
        r = requests.get(f"{API_BASE}/history/list", headers=AUTH_HEADER, timeout=5)
        r.raise_for_status()
        return r.json().get("chat_ids", [])
    except Exception as e:
        print(f"❌ list_chats failed: {e}")
        return []


def delete_chat(chat_id):
    try:
        r = requests.delete(f"{API_BASE}/history/{chat_id}", headers=AUTH_HEADER, timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ delete_chat failed: {e}")
        return False


def load_history(chat_id):
    try:
        r = requests.get(f"{API_BASE}/history/{chat_id}", headers=AUTH_HEADER, timeout=5)
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception as e:
        print(f"❌ load_history failed: {e}")
        return []


def stream_chat(query, chat_id):
    try:
        payload = {"chat_id": chat_id, "query": query}
        with requests.post(
            f"{API_BASE}/chat/stream",
            headers=AUTH_HEADER,
            json=payload,
            stream=True,
            timeout=30,
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as e:
        yield f"❌ stream_chat failed: {e}"


def init_chat(chat_id: str, prompt: str = "Hello!"):
    try:
        payload = {"chat_id": chat_id, "query": prompt}
        r = requests.post(f"{API_BASE}/chat", headers=AUTH_HEADER, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ init_chat failed: {e}")
        return False
