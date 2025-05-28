# app/agent_router.py  – overwrite the file
import asyncio
from functools import lru_cache

from app.llm import get_llm
from app.k8s_agent import get_k8s_agent
from app.fmp_agent import get_fmp_agent
from app.rag_agent import get_rag_chain

_SYS = "You are a router. Reply with exactly one token: K8S, FMP, RAG, or CHAT."
_EXAMPLES = [
    ("Scale deployment payments to 4 replicas", "K8S"),
    ("What is AAPL dividend yield?", "FMP"),
    ("Explain EBITDA in simple terms.", "RAG"),
    ("Hello, how are you?", "CHAT"),
]

@lru_cache(128)
def determine_route(query: str) -> str:
    examples = "\n".join(f"Q: {q}\nA: {a}" for q, a in _EXAMPLES)
    prompt = f"{_SYS}\n{examples}\nQ: {query}\nA:"
    return get_llm().predict(prompt).strip().upper()

async def route_query(query: str, history):
    route = await asyncio.to_thread(determine_route, query)

    if route == "K8S":
        return await asyncio.to_thread(get_k8s_agent().run, query)
    if route == "FMP":
        return await asyncio.to_thread(get_fmp_agent().run, query)
    if route in ("RAG", "CHAT"):
        # CHAT just means “let the LLM answer directly”
        chain = get_rag_chain() if route == "RAG" else get_llm()
        return await asyncio.to_thread(chain.run if hasattr(chain, "run") else chain.predict, query)

    # safety-net
    return "Sorry, I’m not sure how to answer that."
