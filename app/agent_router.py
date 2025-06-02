# app/agent_router.py

"""
Central router that decides which tool answers a user question.

Order of precedence
────────────────────
1.  Hard-coded heuristics  (regexes → Finance / K8s)
2.  Fast LLM classifier    (K8S | FMP | COT_RAG | CHAT)
"""

from __future__ import annotations

import asyncio, json, re
from functools import lru_cache
from typing import Dict, List, Literal

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.llm import get_llm
from app.fmp_agent import get_fmp_agent
from app.k8s_agent import get_k8s_agent
from app.rag_agent import get_cot_rag_chain, run_parallel_searches
from app.memory import save_chat, format_chat_history

# ───────── Finance heuristics ─────────
_FIN_KEYWORDS = re.compile(
    r"\b(price|quote|p/e|eps|dividend|yield|fundamental|roe|market cap|"
    r"fluctuation|performance|last \d+ (day|week|month|year)s?)\b",
    re.I,
)

def _extract_symbol(text: str) -> str | None:
    words = re.findall(r"\b[A-Z]{2,5}\b", text.strip()[1:])
    return words[0] if words else None

def _looks_like_finance(q: str) -> bool:
    return _FIN_KEYWORDS.search(q) is not None or _extract_symbol(q) is not None

# ───────── K8s heuristics ─────────
_K8S_KEYWORDS = re.compile(
    r"\b(scale|restart|pod|deployment|namespace|cpu usage|replica)\b",
    re.I,
)

def _looks_like_k8s(q: str) -> bool:
    return _K8S_KEYWORDS.search(q) is not None

# ───────── Classifier ─────────
@lru_cache(512)
def _classifier(q: str) -> Literal["K8S", "FMP", "COT_RAG", "CHAT"]:
    try:
        prompt = (
            "You are a router. Return exactly one token from this list:\n"
            "K8S, FMP, COT_RAG, CHAT\n\n"
            f"Question: {q}\nAnswer:"
        )
        return get_llm().predict(prompt).strip().upper()
    except Exception as e:
        print(f"⚠️ LLM classifier failed: {e}")
        return "COT_RAG"

# ───────── Planner (CoT) ─────────
_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a step-by-step planner for a multi-hop RAG system. "
     "Break the question into independent retrieval steps followed by reasoning.\n\n"
     "Return ONLY JSON.\n\n"
     "Format:\n"
     '{\n'
     '  "steps": [\n'
     '    {"type": "SEARCH", "query": "<semantic search query>"},\n'
     '    {"type": "SEARCH", "query": "<another search if needed>"},\n'
     '    {"type": "COMPUTE", "instruction": "<how to reason over retrieved results>"}\n'
     '  ]\n'
     '}\n\n'
     "Be strict: never include natural language outside the JSON."),
    ("user", "{question}"),
])
_PLANNER_LLM = get_llm(streaming=False)

# ───────── Public API ─────────
async def route_query(q: str, history: List[Dict], chat_id: str = "default") -> str:
    if _looks_like_finance(q):
        return await asyncio.to_thread(get_fmp_agent().run, q)

    if _looks_like_k8s(q):
        return await asyncio.to_thread(get_k8s_agent().run, q)

    tool = determine_route(q)
    print(f"🧭 Routing decision: {tool}")

    if tool == "FMP":
        return await asyncio.to_thread(get_fmp_agent().run, q)

    if tool == "K8S":
        return await asyncio.to_thread(get_k8s_agent().run, q)

    context = format_chat_history(history)

    if tool == "CHAT":
        prompt = f"{context}\nUser: {q}\nAssistant:" if context else f"User: {q}\nAssistant:"
        return await asyncio.to_thread(get_llm().predict, prompt)

    # ───── COT-RAG starts here ─────
    rag_chain = get_cot_rag_chain()
    raw_plan = "[uninitialized]"

    try:
        plan_response = _PLANNER_LLM.invoke(_PLAN_PROMPT.format(question=q))
        raw_plan = plan_response.content if isinstance(plan_response, AIMessage) else str(plan_response)
        raw_plan = re.sub(r"^```(json)?|```$", "", raw_plan.strip(), flags=re.MULTILINE)
        plan = json.loads(raw_plan)

        search_queries = [
            s["query"] for s in plan.get("steps", [])
            if s.get("type") == "SEARCH"
        ]
    except Exception as e:
        print(f"⚠️ Planner failed: {e}\n🧾 Raw:\n{raw_plan}")

        formatted = format_chat_history(history)
        fallback_prompt = f"{formatted}\nUser: {q}" if formatted else q
        result = await asyncio.to_thread(rag_chain.invoke, {"query": fallback_prompt})
        answer = result["result"]
        save_chat(chat_id, q, answer, scratch=None)
        return answer

    docs = await run_parallel_searches(search_queries, rag_chain.retriever)

    scratch = (
        f"# Chat History\n{context}\n\n"
        f"# Plan\n```json\n{json.dumps(plan, indent=2)}\n```\n"
        f"# Retrieved\n" +
        "\n".join(d.page_content[:600] for d in docs)
    )

    answer_prompt = (
        "Use the plan, history, and retrieved docs to answer. "
        "Be concise and cite like (Doc 1).\n\n"
        f"{scratch}\n\nAnswer:"
    )

    answer = await asyncio.to_thread(rag_chain.llm.predict, answer_prompt)
    save_chat(chat_id, q, answer, scratch)
    return answer

def determine_route(question: str) -> str:
    if _looks_like_finance(question):
        return "FMP"
    if _looks_like_k8s(question):
        return "K8S"

    raw = _classifier(question)
    print(f"🧪 Raw classifier result: {raw}")

    if raw in {"CHAT", "COT_RAG"}:
        return "COT_RAG"

    return raw
