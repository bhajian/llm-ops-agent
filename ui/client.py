import os
import requests

# Base config
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


def delete_chat(chat_id: str):
    try:
        r = requests.delete(f"{API_BASE}/history/{chat_id}", headers=AUTH_HEADER, timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ delete_chat failed: {e}")
        return False


def load_history(chat_id: str):
    try:
        r = requests.get(f"{API_BASE}/history/{chat_id}", headers=AUTH_HEADER, timeout=5)
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception as e:
        print(f"❌ load_history failed: {e}")
        return []


def stream_chat(query: str, chat_id: str):
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


def ingest_file(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        r = requests.post(
            f"{API_BASE}/ingest",
            headers=AUTH_HEADER,
            files=files,
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("message", "Uploaded successfully.")
    except Exception as e:
        return f"❌ Upload failed: {e}"
