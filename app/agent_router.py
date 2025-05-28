# --- app/agent_router.py ---
import asyncio
from typing import List, Dict

from app.rag_agent import get_rag_chain
from app.k8s_agent import get_k8s_agent


async def route_query(query: str, history: List[Dict] | None = None) -> str:
    """
    Decide which agent should answer:
      • Dev-Ops questions → kube/openshift agent
      • everything else   → RAG chain
    """
    lower_q = query.lower()
    if any(kw in lower_q for kw in ("pod", "deployment", "namespace", "openshift", "kubernetes")):
        # LangChain agents are synchronous → run in thread so we can await
        return await asyncio.to_thread(get_k8s_agent().run, query)

    # RetrievalQA is also sync → run in thread
    rag_chain = get_rag_chain()
    if history:
        # naïve concatenation; in production feed through a ConversationBufferMemory
        query = "\n".join(h["query"] + " " + h["response"] for h in history[-3:]) + "\n" + query

    return await asyncio.to_thread(rag_chain.run, query)
