# app/agent_router.py
# ────────────────────────────────────────────────────────────
"""
LLM-based router — backend-agnostic (OpenAI, vLLM, llama.cpp, …).
It looks at **both** recent memory and the current question and picks one of:
CHAT – generic conversation via chat_with_memory
FINANCE – calls FMP agent
COT_RAG – CoT reasoning + retrieval from documents
"""

from __future__ import annotations

import asyncio, os, importlib, warnings
from enum import Enum
from typing import Callable, Dict, List

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from app.llm import get_llm # your unified backend
from app.chat_memory import chat_with_memory
from app.fmp_agent import get_fmp_agent

# ─── dynamically load RAG helper ────────────────────────────
def _first_callable(module: str, *fn_names: str):
    """Helper to dynamically import a callable, falling back if module/function not found."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        return None
    for fn in fn_names:
        obj = getattr(mod, fn, None)
        if callable(obj):
            return obj
    return None

# Assuming app.rag_agent.py provides get_rag_chain
get_rag_chain_from_module = _first_callable("app.rag_agent", "get_rag_chain")

# ─── intent labels ──────────────────────────────────────────
class Intent(str, Enum):
    CHAT = "CHAT"
    COT_RAG = "COT_RAG"
    FINANCE = "FINANCE"

# ─── helpers for handlers ───────────────────────────────────
def _convert_history_to_messages(history: List[dict]) -> List[HumanMessage | AIMessage]:
    """Converts Redis chat history (list of dicts) to LangChain message objects."""
    messages = []
    for item in history:
        if item.get("content"): # Only include messages that have content
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                messages.append(AIMessage(content=item["content"]))
    return messages

# ─── handlers for each intent ───────────────────────────────
async def _handle_chat(q: str, h: List[dict], cid: str) -> str:
    """Handles general conversational queries."""
    # chat_with_memory function is synchronous, so run in a thread
    return await asyncio.to_thread(chat_with_memory, q, h)

async def _handle_finance(q: str, h: List[dict], cid: str) -> str:
    """Handles financial queries using the FMP agent."""
    fmp_agent = get_fmp_agent()
    try:
        # FMP agent's invoke expects a dictionary with 'input' and 'chat_history' (as LangChain messages)
        # AgentExecutor.ainvoke returns a dictionary with an 'output' key
        result = await fmp_agent.ainvoke({"input": q, "chat_history": _convert_history_to_messages(h)})
        return result["output"] if isinstance(result, dict) and "output" in result else str(result)
    except Exception as e:
        warnings.warn(f"FMP Agent failed during ainvoke: {e}")
        return "I had trouble getting financial information. Please try again or rephrase your query."

async def _handle_cot_rag(q: str, h: List[dict], cid: str) -> str:
    """Handles document-based queries using the conversational RAG chain."""
    if get_rag_chain_from_module:
        rag_chain = get_rag_chain_from_module() # Call the factory to get the chain
        try:
            # The RAG chain (create_retrieval_chain) expects 'input' and 'chat_history'
            result = await rag_chain.ainvoke({"input": q, "chat_history": _convert_history_to_messages(h)})
            # The result from create_retrieval_chain has 'answer' and 'context'
            answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)

            print("🔍 RAG retrieved chunks:")
            # 'context' key holds the retrieved documents
            for i, doc in enumerate(result.get("context", [])):
                print(f"📄 [Doc {i+1}] {doc.metadata.get('source', 'Unknown source')}:")
                print(doc.page_content[:300])
                print("---")

            return answer
        except Exception as e:
            warnings.warn(f"COT RAG Agent failed during ainvoke: {e}")
            return "I had trouble understanding and retrieving information from documents. Please try again."
    # Fallback to chat if RAG chain couldn't be loaded or failed
    return await _handle_chat(q, h, cid)

# Map intents to their corresponding handler functions
DEST: Dict[Intent, Callable[[str, List[dict], str], str]] = {
    Intent.CHAT: _handle_chat,
    Intent.FINANCE: _handle_finance,
    Intent.COT_RAG: _handle_cot_rag,
}

# ─── router LLM configuration ───────────────────────────────
_router = get_llm()
if hasattr(_router, "temperature"):
    try:
        _router.temperature = 0.0 # Keep temperature low for deterministic routing
    except Exception:
        pass # Some LLM wrappers might make temperature read-only

# **CRITICAL**: Detailed Router Prompt for accurate LLM routing
_ROUTER_PROMPT = f"""You are an intelligent routing agent. Your task is to analyze the user's message and recent chat history to determine which specialized agent can best handle the request.

Your response MUST be a single word, chosen from the "Available Agents" list below. Do NOT provide any additional text, explanations, or punctuation.

Available Agents:
CHAT: Use for general conversation, greetings, casual chat, or questions that do not require specific tools or document lookup. This is the default if no other agent is clearly appropriate.
FINANCE: Use for any questions specifically related to stock prices, company earnings, financial fundamentals (like P/E ratio, dividends), historical stock data, or information about specific publicly traded companies/tickers. This agent uses financial APIs.
COT_RAG: Use for questions that require retrieving information from documents, knowledge bases, or specific files that have been ingested into the system. This includes questions about specific people (like "who is Behnam Hajian?"), summaries of ingested content, or asking for information that would typically be found in a detailed document.

Consider the user's intent and keywords very carefully. If the user is asking about a person, a document's content, or general knowledge that would be stored in a knowledge base, route to COT_RAG. If it's a financial query, route to FINANCE. Otherwise, route to CHAT.

"""

def _choose_intent(user_msg: str, chat_history: List[dict]) -> Intent:
    """
    Uses the router LLM to choose the intent based on user message and chat history.
    """
    messages_for_router = [
        SystemMessage(content=_ROUTER_PROMPT),
        *_convert_history_to_messages(chat_history), # Pass actual history to the router
        HumanMessage(content=user_msg)
    ]
    try:
        # LLM invocation for routing
        resp = _router.invoke(messages_for_router).content.strip().upper()
        for intent in Intent:
            if resp == intent.value:
                print(f"🧭 Router LLM chose: {resp}") # Log the LLM's routing decision
                return intent
        warnings.warn(f"Router LLM returned unknown intent: '{resp}'. Falling back to CHAT.")
        return Intent.CHAT # Fallback if LLM gives an invalid output
    except Exception as e:
        warnings.warn(f"Router LLM failed to make a decision: {e}. Falling back to CHAT.")
        return Intent.CHAT

# ─── public API (used by main.py) ───────────────────────────
async def route_query(user_msg: str,
                      history: List[dict],
                      chat_id: str) -> str:
    """
    Routes the user's query to the appropriate agent based on LLM intent classification.
    """
    # Determine the intent using the router LLM
    intent = _choose_intent(user_msg, history)
    # Get the handler function for the chosen intent
    handler = DEST[intent]
    # Execute the handler and return its result
    return await handler(user_msg, history, chat_id)
