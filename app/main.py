# app/main.py
# ─────────────────────────────────────────────────────────────
import os, tempfile, asyncio
from pathlib import Path
from typing import AsyncIterator, List, Dict, Any

from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool # Needed for async generator in StreamingResponse

# --- IMPORTS ---\
from app.auth import verify_token
from app.memory import load_chat, save_chat, _r
# FIX: Import the LangGraph app
from app.graph_orchestrator import app as langgraph_app, GraphState # Import the compiled graph and its state type
from app.tools.vector_utils import ingest_file_to_weaviate # For ingest endpoint

app = FastAPI()

# ─────────────────────────────────────────────────────────────
# /chat  (non-stream) - kept for compatibility, but streaming is preferred
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request, user: str = Depends(verify_token)):
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)

    print(f"💬 /chat received query: {query} for chat_id: '{cid}'")

    # 1. Load history
    history = load_chat(cid)

    # 2. Prepare initial state for LangGraph
    initial_state = GraphState(
        user_query=query,
        chat_history=history,
        intent=None,
        ticker_symbol=None,
        finance_tool_output=None,
        rag_condensed_question=None,
        rag_context=None,
        final_answer=None,
        error_message=None
    )

    # 3. Invoke the LangGraph application
    try:
        # LangGraph app.invoke returns the final state
        final_state = await langgraph_app.ainvoke(initial_state)
        answer = final_state.get("final_answer", "I'm sorry, I couldn't generate a response.")
        
        # If there was an error message in the state, prioritize it
        if final_state.get("error_message"):
            answer = f"Error: {final_state['error_message']} {answer}"

        print(f"✅ LangGraph returned answer: {answer}")

    except Exception as e:
        warnings.warn(f"LangGraph execution failed: {e}")
        answer = "I'm sorry, I encountered an unexpected error while processing your request."

    # 4. Save the final answer
    save_chat(cid, query, answer)

    return JSONResponse({"chat_id": cid, "answer": answer})

# ─────────────────────────────────────────────────────────────
# /chat/stream (NEW STREAMING ENDPOINT)
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: Request, user: str = Depends(verify_token)):
    body = await req.json()
    query = body.get("query", "")
    cid = body.get("chat_id", user)

    print(f"⚡️ /chat/stream received query: {query} for chat_id: '{cid}'")

    # 1. Load history
    history = load_chat(cid)
    accumulated_answer_chunks = [] # To save the full answer after streaming

    async def generate_stream():
        # Prepare initial state for LangGraph
        initial_state = GraphState(
            user_query=query,
            chat_history=history,
            intent=None,
            ticker_symbol=None,
            finance_tool_output=None,
            rag_condensed_question=None,
            rag_context=None,
            final_answer=None,
            error_message=None
        )

        try:
            # Astream_log yields events for each step and LLM chunk
            # We are interested in "chunks" from the final synthesis nodes
            async for event in langgraph_app.astream_log(initial_state):
                # LangGraph events have a specific structure.
                # 'ops' contains the state changes. 'chunks' contain streamed content.
                # We want to identify the final answer generation nodes.
                if event.get("event") == "on_chat_model_stream":
                    # Check if the stream event is coming from a 'node' that produces final output
                    # The `name` in `event["name"]` refers to the node name in the graph
                    # The `metadata` can give us info about the parent chain/node
                    
                    # We need to explicitly check if the chunk is from the final LLM invocation.
                    # This typically means checking the `path` or `name` in the event metadata.
                    # The `on_chat_model_stream` events will have `chunk` and `messages`
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, 'content'):
                        # This yields the raw string content for streaming
                        # The client will receive this as plain text.
                        # For production, you might want to wrap this in JSON or SSE format.
                        print(f"🚀 Streaming chunk: {chunk.content}")
                        yield chunk.content
                        accumulated_answer_chunks.append(chunk.content)

        except Exception as e:
            warnings.warn(f"LangGraph streaming execution failed: {e}")
            yield "I'm sorry, I encountered an unexpected error while streaming your request."
        finally:
            # After the stream ends, save the full accumulated answer to chat history
            full_answer = "".join(accumulated_answer_chunks)
            if full_answer: # Only save if there's an answer
                save_chat(cid, query, full_answer)
                print(f"✅ Full streamed answer saved to history for chat_id: '{cid}'")

    # Return as StreamingResponse
    return StreamingResponse(generate_stream(), media_type="text/plain") # text/plain is common for raw streaming


# ─────────────────────────────────────────────────────────────
# /ingest endpoints - (No changes needed beyond import)
# ─────────────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest(file: UploadFile = File(...), user: str = Depends(verify_token)):
    suffix = os.path.splitext(file.filename)[-1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        n_chunks = ingest_file_to_weaviate([tmp_path])
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
async def delete_chat_history(chat_id: str, user: str = Depends(verify_token)):
    _r.delete(f"chat:{chat_id}")
    return {"message": f"Chat history '{chat_id}' deleted."}

