# app/main.py  – overwrite the file
import asyncio
from typing import AsyncIterator

from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

from langchain.callbacks.base import AsyncCallbackHandler

from app.auth import verify_token
from app.memory import load_chat, save_chat
from app.agent_router import determine_route, route_query
from app.llm import get_llm
from app.rag_agent import get_rag_chain

app = FastAPI()

# ---------- non-stream ----------
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body  = await req.json()
    q     = body.get("query", "")
    cid   = body.get("chat_id", user)

    hist  = load_chat(cid)
    ans   = await route_query(q, hist)
    save_chat(cid, q, ans)
    return JSONResponse({"response": ans, "chat_id": cid})

# ---------- stream (RAG only) ----------
class _TokenBuffer(AsyncCallbackHandler):
    def __init__(self):
        self._queue = asyncio.Queue()

    async def on_llm_new_token(self, token, **kw):
        await self._queue.put(token)

    async def aiter(self):
        while True:
            tok = await self._queue.get()
            if tok is None:                 # sentinel
                break
            yield tok

@app.post("/chat/stream")
async def chat_stream(req: Request, user: str = Depends(verify_token)):
    body  = await req.json()
    q     = body.get("query", "")
    cid   = body.get("chat_id", user)
    hist  = load_chat(cid)

    route = await asyncio.to_thread(determine_route, q)

    # ── Not RAG?  → just return single chunk ──────────────────
    if route != "RAG":
        ans = await route_query(q, hist)
        save_chat(cid, q, ans)
        async def once() -> AsyncIterator[str]:
            yield ans
        return StreamingResponse(once(), media_type="text/plain")

    # ── RAG streaming ─────────────────────────────────────────
    buf = _TokenBuffer()
    llm = get_llm(streaming=True, callbacks=[buf])
    rag = get_rag_chain(llm=llm)

    async def token_gen() -> AsyncIterator[str]:
        run_task = asyncio.create_task(asyncio.to_thread(rag.run, q))
        async for tok in buf.aiter():
            yield tok
        ans = await run_task
        save_chat(cid, q, ans)
        await buf._queue.put(None)          # end stream

    return StreamingResponse(token_gen(), media_type="text/plain")
