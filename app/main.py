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
# Helper to check for initial chat trigger phrases
def _is_initial_chat_trigger(query: str, chat_id: str) -> bool:
    """
    Checks if the query indicates a new chat session and there's no existing history.
    """
    normalized_query = query.lower().strip()
    # Check for empty query or common greeting phrases
    if not normalized_query or normalized_query in ["start chat", "hello", "hi", "hey", "help"]:
        # Only consider it an initial trigger if chat history for this ID is empty
        return not load_chat(chat_id)
    return False

# ─────────────────────────────────────────────────────────────
# /chat  (non-stream) - REFACTORED
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)

    print(f"💬 /chat received query: '{query}' for chat_id: '{cid}'")

    # 1. Handle initial empty query or specific start phrases with a static welcome message
    if _is_initial_chat_trigger(query, cid):
        welcome_message = "Hello! How can I help you today?"
        print(f"✅ Returning static welcome message for new chat or initial greeting.")
        return JSONResponse({"response": welcome_message, "chat_id": cid})

    # 2. Load history (only if not an initial empty query handled above)
    history = load_chat(cid)

    # 3. Delegate EVERYTHING to the agent router.
    answer = await route_query(query, history, cid) 

    print(f"✅ Router returned answer: {answer}")

    # 4. Save the final answer
    save_chat(cid, query, answer) 

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
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)
    
    print(f"💬 /chat/stream received query: '{query}' for chat_id: '{cid}'")

    # 1. Handle initial empty query or specific start phrases with a static welcome message
    if _is_initial_chat_trigger(query, cid):
        welcome_message = "Hello! How can I help you today?"
        async def welcome_stream():
            yield welcome_message
        print(f"✅ Returning static welcome message stream for new chat or initial greeting.")
        return StreamingResponse(welcome_stream(), media_type="text/plain")

    history = load_chat(cid)

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
