# app/main.py
# ─────────────────────────────────────────────────────────────
import os, tempfile, asyncio
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler

# --- IMPORTS ---
from app.auth import verify_token
from app.memory import load_chat, save_chat, _r
from app.agent_router import route_query # <--- THIS IS THE ONLY ROUTER IMPORT NOW
from app.tools.vector_utils import ingest_file_to_weaviate # For ingest endpoint

app = FastAPI()

# ─────────────────────────────────────────────────────────────
# /chat  (non-stream) - REFACTORED
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)

    print(f"💬 /chat received query: {query}")

    # 1. Load history
    history = load_chat(cid)

    # 2. Delegate EVERYTHING to the agent router. No more if/else logic.
    # The router is now fully responsible for picking AND running the correct agent.
    answer = await route_query(query, history, cid) # Calls the async route_query directly

    print(f"✅ Router returned answer: {answer}")

    # 3. Save the final answer
    save_chat(cid, query, answer) # Removed scratch as it's not consistently returned by all agents now

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
# /chat/stream (streams all answers that support it) - REFACTORED
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: Request, user: str = Depends(verify_token)):
    # This endpoint would also be simplified to call a streaming-capable router function.
    # We can address the complexities of generic streaming after fixing the core logic.
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)
    history = load_chat(cid)

    # For now, keeping it simple as discussed, but the full streaming solution
    # would involve more complex changes to route_query itself.
    # The non-streaming route_query is called here.
    answer = await route_query(query, history, cid)
    save_chat(cid, query, answer)

    async def once():
        yield answer
    return StreamingResponse(once(), media_type="text/plain")

# ─────────────────────────────────────────────────────────────
# /ingest (PDF / TXT → Weaviate) - (No changes needed)
# ─────────────────────────────────────────────────────────────
def ingest_files(paths: List[str]) -> int:
    return sum(ingest_file_to_weaviate(p) for p in paths)

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
# /history endpoints - (No changes needed)
# ─────────────────────────────────────────────────────────────
@app.get("/history/{chat_id}")
async def get_history(chat_id: str, user: str = Depends(verify_token)):
    raw = load_chat(chat_id)
    sanitized = [{"role": m["role"], "content": m["content"]} for m in raw]
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
