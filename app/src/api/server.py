"""
src/api/server.py
─────────────────
FastAPI entry-point for the multi-agent chatbot.

Routes
------
GET  /healthz
POST /chat             – blocking
POST /chat/stream      – SSE
GET  /history/list
GET  /history/{id}
DELETE /history/{id}
POST /upload           – file upload
"""

from __future__ import annotations

import json
import os # Import os for path operations and directory creation
from typing import Iterator, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File # Import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..graph.graph_builder import handle_chat
from ..integrations.weaviate_schema import ensure_schema
from ..integrations import redis_conn

# ────────────────── FastAPI app ──────────────────
app = FastAPI(
    title="Multi-Agent AI Bot",
    version="0.2.1",
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
    redis_conn.ltrim(_key(cid), -200, -1)


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
    keys = redis_conn.keys("history:*")
    return {"chat_ids": [k.split(":", 1)[1] for k in keys]}


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
    try:
        res = handle_chat(req.message, req.conversation_id, stream=False)

        if not isinstance(res, str):
            res = "".join(
                ck.content if hasattr(ck, "content") else str(ck) for ck in res
            )
        if not res.strip():
            res = "Hello! How can I help you today?"

        _save(req.conversation_id, "user", req.message)
        _save(req.conversation_id, "assistant", res)

        return ChatResponse(answer=res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ────────────────── streaming chat ──────────────
@app.post("/chat/stream", tags=["chat"])
async def chat_stream(raw: Request) -> StreamingResponse:
    """
    SSE endpoint.  We parse JSON manually so *missing* optional
    fields never trigger FastAPI's automatic 422.
    """
    body = await raw.json()
    cid = body.get("conversation_id") or body.get("chat_id")
    msg = body.get("message") or body.get("query")

    if not cid or not msg:
        raise HTTPException(
            status_code=422,
            detail="JSON must contain conversation_id & message",
        )

    def _events() -> Iterator[str]:
        try:
            full = ""
            for chunk in handle_chat(msg, cid, stream=True):
                full += chunk
                yield f"data: {chunk}\n\n"

            if not full.strip():
                full = "Hello! How can I help you today?"
                yield f"data: {full}\n\n"

            yield "data: [DONE]\n\n"

            _save(cid, "user", msg)
            _save(cid, "assistant", full)
        except Exception as exc:
            yield f"data: [ERROR] {str(exc)}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


# ────────────────── file upload route ───────────────
@app.post("/upload", tags=["upload"])
async def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Handles file uploads for ingestion into RAG.
    Saves the file to a temporary location. In a real application,
    you would process this file (e.g., extract text, create embeddings,
    and add to your vector database).
    """
    try:
        # Define a directory to save uploaded files (create if it doesn't exist)
        # In a production environment, you might save to cloud storage or process directly.
        upload_dir = "uploaded_files"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            # Read file in chunks to handle large files efficiently
            while contents := await file.read(1024):
                f.write(contents)

        # IMPORTANT: In a real application, replace this with your RAG ingestion logic.
        # For example, you might call a function like:
        # from ..integrations.document_processor import process_document_for_rag
        # process_document_for_rag(file_path, file.filename)

        return {"message": f"File '{file.filename}' uploaded successfully."}
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")
    