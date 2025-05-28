# app/main.py
import asyncio
from typing import AsyncIterator, List, Dict

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import verify_token
from app.memory import load_chat, save_chat
from app.agent_router import route_query
from app.rag_agent import get_rag_chain
from app.llm import get_llm

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

app = FastAPI()


# ────────────────────────────────────────────────────────────────
# 1) Classic endpoint – returns whole answer as JSON
# ────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: Request, user: str = Depends(verify_token)):
    body      = await request.json()
    query     = body.get("query", "")
    chat_id   = body.get("chat_id", user)        # default: user id

    history   = load_chat(chat_id)
    answer    = await route_query(query, history)

    save_chat(chat_id, query, answer)
    return {"response": answer, "chat_id": chat_id}


# ────────────────────────────────────────────────────────────────
# 2) Streaming endpoint – yields tokens as plain-text chunks
# ────────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request: Request, user: str = Depends(verify_token)):
    body      = await request.json()
    query     = body.get("query", "")
    chat_id   = body.get("chat_id", user)

    history   = load_chat(chat_id)

    # callback handler collects tokens as the LLM generates them
    cb_handler = AsyncIteratorCallbackHandler()

    # tell the LLM to stream
    llm_stream = get_llm().with_options(streaming=True, callbacks=[cb_handler])

    # build a RAG chain that *uses* the streaming LLM
    rag_chain  = get_rag_chain().with_config({"llm": llm_stream})

    async def token_generator() -> AsyncIterator[str]:
        """
        Asynchronously yield tokens to the client while the chain runs.
        """
        # kick off the chain (runs concurrently)
        run_task = asyncio.create_task(
            asyncio.to_thread(rag_chain.run, query if not history else query)
        )

        # forward tokens to the HTTP response as they arrive
        async for token in cb_handler.aiter():
            yield token

        # wait for the chain to finish and persist the full answer
        answer = await run_task
        save_chat(chat_id, query, answer)

    return StreamingResponse(token_generator(), media_type="text/plain")
