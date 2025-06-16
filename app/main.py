# app/main.py
# ─────────────────────────────────────────────────────────────
import os, tempfile, asyncio
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler

# --- INTERNAL IMPORTS ----------------------------------------------------
from app.auth    import verify_token
from app.memory  import load_chat, save_chat, _r
from app.agent_router     import run_agentic_chat
from app.tools.vector_utils import ingest_file_to_weaviate

app = FastAPI()


# ─────────────────────────────────────────────────────────────
# Helper: detect a brand-new chat
# ─────────────────────────────────────────────────────────────
def _is_initial_chat_trigger(query: str, chat_id: str) -> bool:
    norm = query.lower().strip()
    if not norm or norm in {"start chat", "hello", "hi", "hey", "help"}:
        return not load_chat(chat_id)          # history empty?
    return False


# ─────────────────────────────────────────────────────────────
# /chat  – blocking JSON
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body    = await req.json()
    query   = body.get("query", "")
    chat_id = body.get("chat_id", user)

    print(f"💬 /chat → '{query}' (chat_id={chat_id})")

    # 1) persist user message
    await save_chat(chat_id, role="user", msg=query)

    # greeting shortcut
    if _is_initial_chat_trigger(query, chat_id):
        bot_msg = "Hello! How can I help you today?"
        await save_chat(chat_id, role="assistant", msg=bot_msg)
        return JSONResponse({"response": bot_msg, "chat_id": chat_id})

    # 2) delegate to the agent
    answer = await run_agentic_chat(query, chat_id)

    # 3) persist assistant reply
    await save_chat(chat_id, role="assistant", msg=answer)

    return JSONResponse({"response": answer, "chat_id": chat_id})


# ─────────────────────────────────────────────────────────────
# lightweight SSE token buffer (kept for later)
# ─────────────────────────────────────────────────────────────
class _TokenBuffer(AsyncCallbackHandler):
    def __init__(self):
        self._q: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_llm_new_token(self, token, **_):     # streamed tokens
        await self._q.put(token)

    async def on_new_token(self, token, **_):         # legacy hook
        await self._q.put(token)

    # make the handler an async-iterator
    def __aiter__(self) -> AsyncIterator[str]:
        return self._generator()

    async def _generator(self):
        while True:
            tok = await self._q.get()
            if tok is None:
                break
            yield tok


# ─────────────────────────────────────────────────────────────
# /chat/stream  – single-shot SSE
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: Request, user: str = Depends(verify_token)):
    body    = await req.json()
    query   = body.get("query", "")
    chat_id = body.get("chat_id", user)

    print(f"💬 /chat/stream → '{query}' (chat_id={chat_id})")

    # persist user turn
    await save_chat(chat_id, role="user", msg=query)

    # greeting shortcut
    if _is_initial_chat_trigger(query, chat_id):
        async def _greet():
            bot_msg = "Hello! How can I help you today?"
            await save_chat(chat_id, role="assistant", msg=bot_msg)
            yield bot_msg
        return StreamingResponse(_greet(), media_type="text/plain")

    answer = await run_agentic_chat(query, chat_id)
    await save_chat(chat_id, role="assistant", msg=answer)

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"message": f"Ingested {n_chunks} chunks 👍"}


# ─────────────────────────────────────────────────────────────
# /history  – CRUD helpers
# ─────────────────────────────────────────────────────────────
@app.get("/history/{chat_id}")
async def get_history(chat_id: str, user: str = Depends(verify_token)):
    raw = load_chat(chat_id)
    return {
        "chat_id": chat_id,
        "history": [{"role": m["role"], "content": m["msg"]} for m in raw],
    }

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
