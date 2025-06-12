"""
src/api/server.py
─────────────────
FastAPI entry-point for the multi-agent chatbot.

Routes
------
GET  /healthz        – liveness probe
POST /chat           – blocking, returns full answer (JSON)
POST /chat/stream    – SSE streaming endpoint
"""

from __future__ import annotations

from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# internal imports
from ..graph.graph_builder import handle_chat                  # orchestrator
from ..integrations.weaviate_schema import ensure_schema       # schema bootstrap

# ─────────────────── FastAPI app ──────────────────────────────
app = FastAPI(
    title="Multi-Agent AI Bot",
    version="0.1.0",
    description="LangGraph multi-agent chatbot with RAG, MCP and memory.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────── models ───────────────────────────────────
class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    stream: bool = Field(False, description="Return SSE stream if true")


class ChatResponse(BaseModel):
    answer: str


# ─────────────────── startup hook ─────────────────────────────
@app.on_event("startup")
async def _startup() -> None:                  # noqa: D401
    ensure_schema()


# ─────────────────── routes ───────────────────────────────────
@app.get("/healthz", tags=["system"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest) -> ChatResponse:
    """
    Blocking endpoint – waits for the full answer.

    If the orchestrator accidentally returns a generator/iterator,
    consume it and join the chunks so the response model validates.
    """
    try:
        result = handle_chat(req.message, req.conversation_id, stream=False)

        # Coerce generator / iterator → string
        if not isinstance(result, str):
            result = "".join(list(result))

        return ChatResponse(answer=result)
    except Exception as exc:                                   # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Server-Sent Events endpoint (`text/event-stream`).

    Client receives `data: …` lines plus a final `data: [DONE]`.
    """

    def _events() -> Iterator[str]:
        try:
            for chunk in handle_chat(req.message, req.conversation_id, stream=True):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {str(exc)}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")
