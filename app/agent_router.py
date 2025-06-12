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
import json
import re # Import regex module

from enum import Enum
from typing import Callable, Dict, List

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.llm import get_llm
from app.chat_memory import chat_with_memory
# UPDATED IMPORT: Agent modules are now in app/agents/
from app.agents.fmp_agent import get_fmp_agent

# ─── dynamically load RAG helper ────────────────────────────
def _first_callable(module: str, *fn_names: str):
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        return None
    for fn in fn_names:
        obj = getattr(mod, fn, None)
        if callable(obj):
            return obj
    return None

# UPDATED: The module string for dynamic import now points to the new location
get_rag_chain_from_module = _first_callable("app.agents.rag_agent", "get_rag_chain")


# ─── intent labels and Enum for routing output ──────────
class Intent(str, Enum):
    CHAT = "CHAT"
    FINANCE = "FINANCE"
    COT_RAG = "COT_RAG" # CoT = Chain of Thought (for RAG)


_router_llm = get_llm(temperature=0)
_router_string_parser = StrOutputParser()


# Simplified prompt for the LLM router, as it's now a fallback
_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert at classifying user intent. "
        "Your task is to determine if the user's query is purely conversational ('CHAT'). "
        "Only output 'CHAT' if no other specific intent is obvious from the query or history. "
        "DO NOT output 'FINANCE' or 'COT_RAG'. Those will be handled by specific rules."
        "You MUST output **ONLY** the keyword 'CHAT' (no other text, no punctuation, no explanations, no special tokens)."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
])


def _choose_intent(user_msg: str, chat_history: List[dict]) -> Intent:
    normalized_user_msg = user_msg.lower().strip()

    # --- Rule-Based Routing (Prioritized) ---
    # Rule 1: RAG queries (e.g., "who is X", "tell me about Y", if it's a person/concept likely in docs)
    # The 'rastgoo.pdf' contains "Milo Rastgoo"
    if "who is" in normalized_user_msg or \
       "tell me about" in normalized_user_msg or \
       "information about" in normalized_user_msg or \
       "summarize" in normalized_user_msg: # Added 'summarize' for RAG
        # Check if the query contains names or terms likely found in ingested documents
        # This is a heuristic and might need refinement based on your document types
        if "behnam hajian" in normalized_user_msg or \
           "milo rastgoo" in normalized_user_msg or \
           "project" in normalized_user_msg or \
           "report" in normalized_user_msg or \
           "document" in normalized_user_msg or \
           "company" in normalized_user_msg:
            print(f"🧭 Rule-based router chose: COT_RAG for query: '{user_msg}'")
            return Intent.COT_RAG

    # Rule 2: Finance queries (explicit financial terms, stock, price, company name in financial context, date/time)
    finance_keywords = ["stock", "price", "company value", "market cap", "earnings", "dividend", "financials", "analyst", "target"]
    if any(keyword in normalized_user_msg for keyword in finance_keywords) or \
       "date today" in normalized_user_msg or "current time" in normalized_user_msg or \
       "what time is it" in normalized_user_msg or "what date is it" in normalized_user_msg:
        print(f"🧭 Rule-based router chose: FINANCE for query: '{user_msg}'")
        return Intent.FINANCE
    
    # --- Fallback to LLM Router for General CHAT ---
    # If no rule matches, let the LLM decide if it's general CHAT.
    # We explicitly tell the LLM to ONLY output 'CHAT'.
    messages_for_router = _ROUTER_PROMPT.format_messages(
        input=user_msg,
        chat_history=_convert_history_to_messages(chat_history)
    )
    llm_response = None
    raw_content = "N/A" 
    try:
        llm_response = _router_llm.invoke(messages_for_router)
        
        if hasattr(llm_response, 'content'):
            raw_content = llm_response.content.strip()
        else:
            raw_content = str(llm_response).strip()

        # Simple cleaning and parsing for the expected 'CHAT' keyword
        cleaned_keyword = raw_content.split()[0].upper() if raw_content else ""

        if cleaned_keyword == "CHAT":
            print(f"🧭 LLM router chose: CHAT (from raw response: '{raw_content}') for query: '{user_msg}'")
            return Intent.CHAT
        else:
            warnings.warn(
                f"LLM router produced unexpected keyword: '{cleaned_keyword}'. Raw response: '{raw_content}'. "
                "Falling back to CHAT."
            )
            return Intent.CHAT
    except Exception as e:
        warnings.warn(f"Router LLM (fallback) failed: {e}. Raw response: '{raw_content}'. Falling back to CHAT.")
        return Intent.CHAT

# ─── public API (used by main.py) ───────────────────────────
async def route_query(user_msg: str,
                      history: List[dict],
                      chat_id: str) -> str:
    """
    Routes the user's query to the appropriate agent based on LLM decision.
    """
    # 1. Choose intent via Rule-based or LLM-based router
    intent = _choose_intent(user_msg, history)

    # 2. Call appropriate agent based on intent
    answer = ""
    if intent == Intent.CHAT:
        answer = chat_with_memory(user_msg, history)
    elif intent == Intent.FINANCE:
        try:
            fmp_agent = get_fmp_agent()
            # Pass streaming=True for async operations
            if hasattr(fmp_agent, "ainvoke"):
                result = await fmp_agent.ainvoke({"input": user_msg, "chat_history": _convert_history_to_messages(history)}, config={"callbacks": [], "tags": ["fmp_agent"], "max_retries": 1})
            else:
                result = await asyncio.to_thread(fmp_agent.invoke, {"input": user_msg, "chat_history": _convert_history_to_messages(history)})
            answer = result.get("output", "Could not get an answer from the FMP agent.")
        except Exception as e:
            warnings.warn(f"FMP Agent failed during ainvoke: {e}")
            answer = "I'm sorry, I encountered an error while trying to get financial data."
    elif intent == Intent.COT_RAG:
        if get_rag_chain_from_module:
            try:
                # FIX: Pass streaming=True to get_rag_chain_from_module
                rag_chain = get_rag_chain_from_module(streaming=True) 
                if hasattr(rag_chain, "ainvoke"):
                    result = await rag_chain.ainvoke({"input": user_msg, "chat_history": _convert_history_to_messages(history)}, config={"callbacks": [], "tags": ["rag_agent"], "max_retries": 1})
                else:
                    result = await asyncio.to_thread(rag_chain.invoke, {"input": user_msg, "chat_history": _convert_history_to_messages(history)})
                answer = result.get("answer", "I could not find relevant information in the documents.")
            except Exception as e:
                warnings.warn(f"RAG Agent failed during ainvoke: {e}")
                answer = "I'm sorry, I encountered an error while trying to retrieve information from documents."
        else:
            warnings.warn("RAG agent module 'app.agents.rag_agent' or 'get_rag_chain' not found. Falling back to CHAT.")
            answer = chat_with_memory(user_msg, history) 
    else:
        answer = chat_with_memory(user_msg, history)
        warnings.warn(f"Unhandled intent: {intent}. Falling back to CHAT.")
    
    return answer

# ───────────────── helpers ───────────────────────────────────────────────
def _convert_history_to_messages(history: List[dict]) -> List[AIMessage | HumanMessage]:
    """Converts the raw history dicts to LangChain message objects."""
    converted_messages = []
    for entry in history:
        if entry["role"] == "user":
            converted_messages.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            # For assistant messages, ensure we only take content, not scratch
            converted_messages.append(AIMessage(content=entry["content"]))
    return converted_messages
