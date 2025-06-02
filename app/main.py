# app/main.py

import os, tempfile, asyncio
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler

from app.auth import verify_token
from app.memory import load_chat, save_chat, format_chat_history, _r
from app.agent_router import determine_route, route_query
from app.llm import get_llm
from app.rag_agent import get_cot_rag_chain
from app.tools.vector_utils import ingest_file_to_weaviate

app = FastAPI()


# ─────────────────────────────────────────────────────────────
# /chat  (non-stream)
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)

    print(f"💬 /chat received query: {query}")

    history = load_chat(cid)
    route = await asyncio.to_thread(determine_route, query)
    print(f"🧭 Routing decision: {route}")

    if route != "RAG":
        print("⚙️ Handling via non-RAG toolchain")
        answer = await route_query(query, history, cid)
        scratch = None
    else:
        print("🧠 Handling with COT_RAG")
        rag_chain = get_cot_rag_chain()
        formatted_context = format_chat_history(history)
        full_query = f"{formatted_context}\nUser: {query}" if formatted_context else query
        result = rag_chain.invoke({"query": full_query})
        answer = result["result"]
        scratch = result.get("scratchpad") or None

        print("🔍 RAG retrieved chunks:")
        for i, doc in enumerate(result.get("source_documents", [])):
            print(f"📄 [Doc {i+1}] {doc.metadata.get('source', '')}:")
            print(doc.page_content[:300])
            print("---")

        print(f"🧠 RAG Answer: {answer}")

    save_chat(cid, query, answer, scratch)
    return JSONResponse({"response": answer, "chat_id": cid})


# ─────────────────────────────────────────────────────────────
# token buffer for streaming mode
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
# /chat/stream
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: Request, user: str = Depends(verify_token)):
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)

    history = load_chat(cid)
    route = await asyncio.to_thread(determine_route, query)

    if route != "RAG":
        answer = await route_query(query, history, cid)
        save_chat(cid, query, answer)

        async def once():
            yield answer
        return StreamingResponse(once(), media_type="text/plain")

    buf = _TokenBuffer()
    llm = get_llm(streaming=True, callbacks=[buf])
    rag_chain = get_cot_rag_chain(streaming=True)

    formatted_context = format_chat_history(history)
    full_query = f"{formatted_context}\nUser: {query}" if formatted_context else query

    async def gen() -> AsyncIterator[str]:
        run_task = asyncio.create_task(
            asyncio.to_thread(rag_chain.invoke, {"query": full_query})
        )

        async for token in buf.aiter():
            yield token

        result = await run_task
        answer = result["result"]
        scratch = result.get("scratchpad") or None

        save_chat(cid, query, answer, scratch)
        await buf._q.put(None)

    return StreamingResponse(gen(), media_type="text/plain")


# ─────────────────────────────────────────────────────────────
# /ingest (PDF / TXT → Weaviate)
# ─────────────────────────────────────────────────────────────
def ingest_files(paths: List[str]) -> int:
    total = 0
    for path in paths:
        total += ingest_file_to_weaviate(path)
    return total


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
        Path(tmp_path).unlink(missing_ok=True)

    return {"message": f"Ingested {n_chunks} chunks 👍"}


# ─────────────────────────────────────────────────────────────
# /history endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/history/{chat_id}")
async def get_history(chat_id: str, user: str = Depends(verify_token)):
    raw = load_chat(chat_id)
    sanitized = [{"role": m["role"], "content": m["content"]} for m in raw]
    return {"chat_id": chat_id, "history": sanitized}


@app.get("/history/list")
async def list_chat_ids(user: str = Depends(verify_token)):
    keys = _r.keys("chat:*")
    chat_ids = [k.split(":")[1] for k in keys]
    return {"chat_ids": sorted(chat_ids)}


@app.delete("/history/{chat_id}")
async def delete_history(chat_id: str, user: str = Depends(verify_token)):
    deleted = _r.delete(f"chat:{chat_id}")
    if deleted:
        return {"message": f"Deleted chat history for '{chat_id}'."}
    return {"message": f"No history found for '{chat_id}'."}
