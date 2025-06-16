# app/main.py
# ─────────────────────────────────────────────────────────────
import os, tempfile, asyncio
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler

# --- INTERNAL IMPORTS ---
from app.auth import verify_token
from app.memory import load_chat, _r               # save_chat now handled inside agent
from app.agent_router import run_agentic_chat      # new orchestrator
from app.tools.vector_utils import ingest_file_to_weaviate

app = FastAPI()


# ─────────────────────────────────────────────────────────────
# Helper: detect a brand-new chat
# ─────────────────────────────────────────────────────────────
def _is_initial_chat_trigger(query: str, chat_id: str) -> bool:
    """
    True if the user sent an empty / greeting message AND no history exists.
    """
    normalized = query.lower().strip()
    if not normalized or normalized in {"start chat", "hello", "hi", "hey", "help"}:
        return not load_chat(chat_id)
    return False


# ─────────────────────────────────────────────────────────────
# /chat  – blocking JSON
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body   = await req.json()
    query  = body.get("query", "")
    chat_id = body.get("chat_id", user)

    print(f"💬 /chat → '{query}' (chat_id={chat_id})")

    # Greeting / new-thread shortcut
    if _is_initial_chat_trigger(query, chat_id):
        return JSONResponse({"response": "Hello! How can I help you today?",
                             "chat_id": chat_id})

    # Delegate to the autonomous agent (handles memory internally)
    answer = await run_agentic_chat(query, chat_id)
    return JSONResponse({"response": answer, "chat_id": chat_id})


# ─────────────────────────────────────────────────────────────
# Token buffer (kept for future real-time streaming)
# ─────────────────────────────────────────────────────────────
class _TokenBuffer(AsyncCallbackHandler):
    def __init__(self):
        self._q: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_llm_new_token(self, token, **kwargs):
        await self._q.put(token)

    async def on_new_token(self, token, **kwargs):
        await self._q.put(token)

    async def aiter(self) -> AsyncIterator[str]:
        while True:
            tok = await self._q.get()
            if tok is None:
                break
            yield tok


# ─────────────────────────────────────────────────────────────
# /chat/stream  – Server-Sent Events lite
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: Request, user: str = Depends(verify_token)):
    body   = await req.json()
    query  = body.get("query", "")
    chat_id = body.get("chat_id", user)

    print(f"💬 /chat/stream → '{query}' (chat_id={chat_id})")

    # Greeting / new-thread shortcut
    if _is_initial_chat_trigger(query, chat_id):
        async def _greet_stream():
            yield "Hello! How can I help you today?"
        return StreamingResponse(_greet_stream(), media_type="text/plain")

    answer = await run_agentic_chat(query, chat_id)

    async def _once():
        yield answer
    return StreamingResponse(_once(), media_type="text/plain")


# ─────────────────────────────────────────────────────────────
# /ingest  – PDF/TXT → Weaviate
# ─────────────────────────────────────────────────────────────
def _ingest_files(paths: List[str]) -> int:
    return sum(ingest_file_to_weaviate(p) for p in paths)

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), user: str = Depends(verify_token)):
    suffix = os.path.splitext(file.filename)[-1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        n_chunks = _ingest_files([tmp_path])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"message": f"Ingested {n_chunks} chunks 👍"}


# ─────────────────────────────────────────────────────────────
# /history  – CRUD helpers
# ─────────────────────────────────────────────────────────────
@app.get("/history/{chat_id}")
async def get_history(chat_id: str, user: str = Depends(verify_token)):
    raw = load_chat(chat_id)
    sanitized = [{"role": m["role"], "content": m["msg"]} for m in raw]
    return {"chat_id": chat_id, "history": sanitized}

@app.get("/history/list")
async def list_chat_ids(user: str = Depends(verify_token)):
    keys = _r.keys("chat:*")
    return {"chat_ids": sorted(k.split(":")[1] for k in keys)}

@app.delete("/history/{chat_id}")
async def delete_history(chat_id: str, user: str = Depends(verify_token)):
    deleted = _r.delete(f"chat:{chat_id}")
    if deleted:
        return {"message": f"Deleted chat history for '{chat_id}'."}
    return {"message": f"No history found for '{chat_id}'."}
