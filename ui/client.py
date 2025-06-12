"""
ui/client.py
============
Tiny synchronous wrapper around the FastAPI backend.  The backend must expose:

POST /chat               {"conversation_id": str, "message": str}
POST /chat/stream        same payload, returns Server-Sent Events
GET  /history/list       {"chat_ids":[...]}
GET  /history/{id}       {"history":[...]}
DELETE /history/{id}
"""

from __future__ import annotations

import os
import requests

BACKEND = os.getenv("BACKEND_URL", "http://agent-server:8000")

# ───────── chat helpers ─────────
def chat_blocking(conversation_id: str, message: str) -> str:
    r = requests.post(
        f"{BACKEND}/chat",
        json={"conversation_id": conversation_id, "message": message},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["answer"]


def chat_stream(conversation_id: str, message: str):
    """Yield raw SSE lines (bytes)."""
    print("PAYLOAD →", {"conversation_id": conversation_id, "message": message}, flush=True)
    r = requests.post(
        f"{BACKEND}/chat/stream",
        json={"conversation_id": conversation_id, "message": message},
        stream=True,
        timeout=60,
    )
    r.raise_for_status()
    return r.iter_lines()


# old pages import `stream_chat`
stream_chat = chat_stream

# ───────── history helpers ─────────
def list_history() -> list[str]:
    return requests.get(f"{BACKEND}/history/list", timeout=30).json()["chat_ids"]


def get_history(conversation_id: str) -> list[dict]:
    return requests.get(f"{BACKEND}/history/{conversation_id}", timeout=30).json()["history"]


def delete_history(conversation_id: str) -> None:
    requests.delete(f"{BACKEND}/history/{conversation_id}", timeout=30).raise_for_status()


# ───────── optional upload helper ─────────
def ingest_file(file) -> str:
    r = requests.post(f"{BACKEND}/upload", files={"file": (file.name, file.read())})
    if r.status_code == 404:
        return "Upload route not implemented on server."
    r.raise_for_status()
    return r.json().get("message", "Uploaded successfully!")
