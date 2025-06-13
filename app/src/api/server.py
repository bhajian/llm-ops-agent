"""
server.py
---------
FastAPI entry-point for the multi-agent chatbot.

Routes
------
GET  /healthz
POST /chat             – blocking
POST /chat/stream      – SSE
GET  /history/list
GET  /history/{id}
DELETE /history/{id}
POST /upload           – file upload (saved to ./uploaded_files/)
"""

from __future__ import annotations
import json
import os
import time
from typing import AsyncIterator, Iterator, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..graph.graph_builder import handle_chat
from ..integrations.weaviate_schema import ensure_schema
from ..integrations import redis_conn

# ────────────────── FastAPI app ──────────────────
app = FastAPI(
    title="Multi-Agent AI Bot",
    version="0.3.1",
    description="LangGraph chatbot with Redis-backed history.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────── Pydantic models ──────────────
class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    stream: bool | None = False


class ChatResponse(BaseModel):
    answer: str


# ────────────────── startup ──────────────────────
@app.on_event("startup")
async def _startup() -> None:  # noqa: D401
    ensure_schema()


# ────────────────── Redis helpers ────────────────
def _key(cid: str) -> str:
    return f"history:{cid}"


def _save(cid: str, role: str, content: str) -> None:
    redis_conn.rpush(_key(cid), json.dumps({"role": role, "content": content}))
    redis_conn.ltrim(_key(cid), -200, -1)  # keep last 200 messages


def _load(cid: str) -> List[dict]:
    raw = redis_conn.lrange(_key(cid), 0, -1) or []
    return [json.loads(r) for r in raw]


# ────────────────── system route ────────────────
@app.get("/healthz", tags=["system"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ────────────────── history API ────────────────
@app.get("/history/list", tags=["history"])
def list_history() -> dict[str, List[str]]:
    try:
        keys = redis_conn.keys("history:*")
        ids = [k.split(":", 1)[1] for k in keys]
    except Exception:
        ids = []  # Redis unavailable → return empty list; UI keeps working
    return {"chat_ids": ids}


@app.get("/history/{chat_id}", tags=["history"])
def get_history(chat_id: str) -> dict[str, List[dict]]:
    return {"history": _load(chat_id)}


@app.delete("/history/{chat_id}", tags=["history"])
def delete_history(chat_id: str) -> dict[str, str]:
    redis_conn.delete(_key(chat_id))
    return {"status": "deleted"}


# ────────────────── blocking chat ───────────────
@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest) -> ChatResponse:
    res = handle_chat(req.message, req.conversation_id, stream=False)

    if not isinstance(res, str):
        # safety: convert any iterable (should not happen in blocking mode)
        res = "".join(chunk.content if hasattr(chunk, "content") else str(chunk) for chunk in res)

    res = res.strip() or "Hello! How can I help you today?"

    _save(req.conversation_id, "user", req.message)
    _save(req.conversation_id, "assistant", res)

    return ChatResponse(answer=res)


# ────────────────── streaming chat ──────────────
@app.post("/chat/stream", tags=["chat"])
async def chat_stream(raw: Request) -> StreamingResponse:
    """
    SSE endpoint. We parse JSON manually so *missing* optional
    fields never trigger FastAPI's automatic 422.
    """
    body = await raw.json()
    cid = body.get("conversation_id") or body.get("chat_id")
    msg = body.get("message") or body.get("query")

    if not cid or not msg:
        raise HTTPException(422, "JSON must contain conversation_id & message")

    async def _events() -> AsyncIterator[str]:
        user_saved = False
        try:
            async for chunk in handle_chat(msg, cid, stream=True):
                if not user_saved:
                    _save(cid, "user", msg)
                    user_saved = True
                if chunk:
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
            # store assistant reply as a single string
            _save(cid, "assistant", "(stream)")  # placeholder; update below
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


# ────────────────── file upload route ───────────
@app.post("/upload", tags=["upload"])
async def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Handles file uploads for ingestion into RAG.
    Saves the file to ./uploaded_files/.
    """
    try:
        upload_dir = "uploaded_files"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024):
                f.write(chunk)

        # TODO: process file for RAG if needed
        return {"message": f"File '{file.filename}' uploaded successfully."}
    except Exception as e:
        raise HTTPException(500, f"Error uploading file: {e}") from e
