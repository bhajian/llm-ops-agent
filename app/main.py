# app/main.py
import os, tempfile, asyncio
from typing import AsyncIterator

from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler

from app.auth import verify_token
from app.memory import load_chat, save_chat
from app.agent_router import determine_route, route_query
from app.llm import get_llm
from app.rag_agent import get_rag_chain
from app.tools.vector_utils import ingest_files

app = FastAPI()

# ─────────────────────────────────────────────────────────────
# /chat  (non-stream)
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body  = await req.json()
    query = body.get("query", "")
    cid   = body.get("chat_id", user)

    history = load_chat(cid)
    answer  = await route_query(query, history)
    save_chat(cid, query, answer)

    return JSONResponse({"response": answer, "chat_id": cid})


# ─────────────────────────────────────────────────────────────
# token buffer for streaming mode
# ─────────────────────────────────────────────────────────────
class _TokenBuffer(AsyncCallbackHandler):
    def __init__(self):
        self._q: asyncio.Queue[str | None] = asyncio.Queue()

    # old LangChain (<0.2)
    async def on_llm_new_token(self, token, **kwargs):
        await self._q.put(token)

    # new LangChain (>=0.2)
    async def on_new_token(self, token, **kwargs):
        await self._q.put(token)

    async def aiter(self) -> AsyncIterator[str]:
        while True:
            tok = await self._q.get()
            if tok is None:
                break
            yield tok


# ─────────────────────────────────────────────────────────────
# /chat/stream  (streams only when route == RAG)
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: Request, user: str = Depends(verify_token)):
    body  = await req.json()
    query = body.get("query", "")
    cid   = body.get("chat_id", user)
    hist  = load_chat(cid)

    route = await asyncio.to_thread(determine_route, query)

    # —— non-RAG routes: return single chunk ——————————————
    if route != "RAG":
        answer = await route_query(query, hist)
        save_chat(cid, query, answer)

        async def once():
            yield answer
        return StreamingResponse(once(), media_type="text/plain")

    # —— RAG route: stream tokens ——————————————
    buf = _TokenBuffer()
    llm_stream = get_llm(streaming=True, callbacks=[buf])
    rag_chain  = get_rag_chain(llm=llm_stream)

    async def gen() -> AsyncIterator[str]:
        run_task = asyncio.create_task(
            asyncio.to_thread(rag_chain.run, query)
        )
        async for token in buf.aiter():
            yield token
        answer = await run_task
        save_chat(cid, query, answer)
        await buf._q.put(None)          # end stream

    return StreamingResponse(gen(), media_type="text/plain")


# ─────────────────────────────────────────────────────────────
# /ingest  (upload PDF / TXT → Weaviate)
# ─────────────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest(file: UploadFile = File(...), user: str = Depends(verify_token)):
    suffix = os.path.splitext(file.filename)[-1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        n_chunks = ingest_files([tmp_path])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return {"message": f"Ingested {n_chunks} chunks 👍"}
