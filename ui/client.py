# ui/client.py
import os, requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://agent-server:8000")
AUTH_TOKEN  = os.getenv("AUTH_TOKEN",  "supersecrettoken")
HEADERS     = {"Authorization": f"Bearer {AUTH_TOKEN}"}


def stream_chat(query: str, chat_id: str):
    payload = {"query": query, "chat_id": chat_id}
    with requests.post(f"{BACKEND_URL}/chat/stream",
                       json=payload, headers=HEADERS,
                       stream=True, timeout=300) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def ingest_file(filepath: str):
    with open(filepath, "rb") as f:
        files = {"file": (os.path.basename(filepath), f)}
        r = requests.post(f"{BACKEND_URL}/ingest",
                          headers=HEADERS, files=files, timeout=60)
    r.raise_for_status()
    return r.json().get("message", "Uploaded")
