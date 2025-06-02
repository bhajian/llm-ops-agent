# app/agent_router.py
import asyncio, json, logging, re
from functools import lru_cache
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from app.llm import get_llm
from app.k8s_agent import get_k8s_agent
from app.fmp_agent import get_fmp_agent
from app.rag_agent import get_cot_rag_chain, run_parallel_searches
from app.memory import save_chat


# ───────────────────────────────────────────────
# 0.  FAST CLASSIFIER  (K8S | FMP | COT_RAG | CHAT)
# ───────────────────────────────────────────────
@lru_cache(maxsize=256)
def _tool_route(question: str) -> str:
    prompt = (
        "You are a classifier.  Reply with ONE token exactly:\n"
        "K8S, FMP, COT_RAG, or CHAT.\n\n"
        f"Question: {question}\nAnswer:"
    )
    return get_llm().predict(prompt).strip().upper()


# compatibility shim for old imports
def determine_route(question: str) -> str:      # <── main.py expects this
    return _tool_route(question)


# ───────────────────────────────────────────────
# 1.  PLANNER  (creates JSON plan for COT_RAG)
# ───────────────────────────────────────────────
_PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Convert the user's question into a JSON plan.\n"
            "Format: {steps:[{type:'SEARCH', query:string}|"
            "{type:'COMPUTE', formula:string}, ...]}.\n"
            "Return ONLY json, no commentary."
        ),
        ("user", "{question}"),
    ]
)
_PLANNER_LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ───────────────────────────────────────────────
# 2.  MAIN ENTRY
# ───────────────────────────────────────────────
async def route_query(question: str, history: List[Dict]) -> str:
    route = await asyncio.to_thread(_tool_route, question)

    # ---------- direct tool routes ----------
    if route == "K8S":
        return await asyncio.to_thread(get_k8s_agent().run, question)

    if route == "FMP":
        return await asyncio.to_thread(get_fmp_agent().run, question)

    # ---------- plain chat ----------
    if route == "CHAT":
        ctx = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history[-10:])
        prompt = f"{ctx}\nUser: {question}\nAssistant:" if ctx else question
        return await asyncio.to_thread(get_llm().predict, prompt)

    # ---------- CoT-RAG ----------
    ## 2-a  plan
    try:
        plan_json = _PLANNER_LLM.predict(
            _PLANNER_PROMPT.format(question=question)
        )
        plan = json.loads(plan_json)
    except Exception as e:
        logging.warning("Planner failed (%s) – falling back to simple RAG", e)
        rag_chain = get_cot_rag_chain()
        return await asyncio.to_thread(rag_chain.run, question)

    ## 2-b  parallel retrieval
    search_qs = [
        s["query"] for s in plan.get("steps", []) if s.get("type") == "SEARCH"
    ]
    rag_chain = get_cot_rag_chain()           # build once
    docs = await run_parallel_searches(search_qs, rag_chain.retriever)

    ## 2-c  assemble scratch-pad (not shown to user)
    scratch = (
        f"# Plan\n```json\n{plan_json}\n```\n"
        f"# Retrieved\n" + "\n".join(d.page_content[:600] for d in docs)
    )

    ## 2-d  final answer
    answer_prompt = (
        "You are an analyst.  Use the plan and retrieved docs below to answer.\n"
        "Be concise and cite sources like (Doc 1).\n\n"
        f"{scratch}\n\nAnswer:"
    )
    answer_llm = rag_chain.combine_documents_chain.llm
    answer = await asyncio.to_thread(answer_llm.predict, answer_prompt)

    # store chain-of-thought in memory (hidden from user)
    save_chat(plan.get("chat_id", "default"), question, answer, scratch)
    return answer
