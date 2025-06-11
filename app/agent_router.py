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
import json

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.llm import get_llm
from app.chat_memory import chat_with_memory
from app.fmp_agent import get_fmp_agent

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

get_rag_chain_from_module = _first_callable("app.rag_agent", "get_rag_chain")

# ─── intent labels and Pydantic for routing output ──────────
class Intent(str, Enum):
    CHAT = "CHAT"
    COT_RAG = "COT_RAG"
    FINANCE = "FINANCE"

class RouterOutput(BaseModel):
    """Output schema for the router LLM, specifying the chosen intent."""
    intent: Intent = Field(..., description="The recognized intent of the user's query.")

_router_parser = PydanticOutputParser(pydantic_object=RouterOutput)

# **FIX FOR KeyError: '"intent"' (and previous schema-related KeyErrors)**
# Manually generate the JSON schema string once.
# This string will be embedded as a literal in the prompt template.
_router_output_schema_json_string = ""
try:
    _router_output_schema_json_string = json.dumps(RouterOutput.model_json_schema(), indent=2)
except AttributeError:
    try:
        _router_output_schema_json_string = json.dumps(RouterOutput.schema(), indent=2)
    except AttributeError as e:
        warnings.warn(f"Could not generate JSON schema string from RouterOutput: {e}. Output parser instructions might be missing or incorrect.")
        _router_output_schema_json_string = "{}" # Fallback to empty schema if all fails


# ─── helpers for handlers ───────────────────────────────────
def _convert_history_to_messages(history: List[dict]) -> List[HumanMessage | AIMessage]:
    messages = []
    for item in history:
        if item.get("content"):
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                messages.append(AIMessage(content=item["content"]))
    return messages

# ─── handlers for each intent ───────────────────────────────
async def _handle_chat(q: str, h: List[dict], cid: str) -> str:
    return await asyncio.to_thread(chat_with_memory, q, h)

async def _handle_finance(q: str, h: List[dict], cid: str) -> str:
    fmp_agent = get_fmp_agent()
    try:
        result = await fmp_agent.ainvoke({"input": q, "chat_history": _convert_history_to_messages(h)})
        return result["output"] if isinstance(result, dict) and "output" in result else str(result)
    except Exception as e:
        warnings.warn(f"FMP Agent failed during ainvoke: {e}")
        return "I had trouble getting financial information. Please try again or rephrase your query."

async def _handle_cot_rag(q: str, h: List[dict], cid: str) -> str:
    if get_rag_chain_from_module:
        rag_chain = get_rag_chain_from_module()
        try:
            result = await rag_chain.ainvoke({"input": q, "chat_history": _convert_history_to_messages(h)})
            answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)

            print("🔍 RAG retrieved chunks:")
            for i, doc in enumerate(result.get("context", [])):
                print(f"📄 [Doc {i+1}] {doc.metadata.get('source', 'Unknown source')}:")
                print(doc.page_content[:300])
                print("---")

            return answer
        except Exception as e:
            warnings.warn(f"COT RAG Agent failed during ainvoke: {e}")
            return "I had trouble understanding and retrieving information from documents. Please try again."
    return await _handle_chat(q, h, cid)

DEST: Dict[Intent, Callable[[str, List[dict], str], str]] = {
    Intent.CHAT: _handle_chat,
    Intent.FINANCE: _handle_finance,
    Intent.COT_RAG: _handle_cot_rag,
}

# ─── router LLM configuration ───────────────────────────────
_router_llm = get_llm()
if hasattr(_router_llm, "temperature"):
    try:
        _router_llm.temperature = 0.0
    except Exception:
        pass

_robust_router_parser = OutputFixingParser.from_llm(parser=_router_parser, llm=_router_llm)

# **CRITICAL**: Router Prompt designed for strict JSON output
# Using separate SystemMessages to avoid formatting conflicts with JSON schema
_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an intelligent routing agent. Your task is to analyze the user's message and recent chat history to determine which specialized agent should handle the request."
        "\n\n---"
        "\nAVAILABLE AGENTS:"
        "\n- CHAT: Use for general conversation, greetings, casual chat, or questions that do not require specific tools or document lookup. This is the default if no other agent is clearly appropriate."
        "\n- FINANCE: Use for any questions specifically related to stock prices, company earnings, financial fundamentals (like P/E ratio, dividends), historical stock data, or information about specific publicly traded companies/tickers. This agent uses financial APIs."
        "\n- COT_RAG: Use for questions that require retrieving information from documents, knowledge bases, or specific files that have been ingested into the system. This includes questions about specific people, summaries of ingested content, or asking for information that would typically be found in a detailed document."
        "\n\n---"
        "\nOUTPUT INSTRUCTIONS:"
        "\nYour response MUST be a JSON object with a single key 'intent'."
        "\nThe value of 'intent' MUST be one of the AVAILABLE AGENTS (CHAT, FINANCE, COT_RAG)."
        "\nDO NOT include any other text, explanations, apologies, or punctuation outside the JSON object."
        "\n\nExample Valid Output:"
        "\n```json"
        "\n{\"intent\": \"FINANCE\"}"
        "\n```"
        "\nConsider the user's intent very carefully. Be decisive."
     )
    ),
    # THIS IS THE FINAL CRITICAL CHANGE: The JSON schema as a completely separate SystemMessage.
    # It ensures the schema string is passed literally to the LLM without any
    # string.format() interference from the main template.
    SystemMessage(content=f"Here is the JSON schema you must adhere to:\n```json\n{_router_output_schema_json_string}\n```"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"), # Ensure human message also uses template variable
])


def _choose_intent(user_msg: str, chat_history: List[dict]) -> Intent:
    messages_for_router = _ROUTER_PROMPT.format_messages(
        input=user_msg,
        chat_history=_convert_history_to_messages(chat_history)
    )
    llm_response = None
    try:
        llm_response = _router_llm.invoke(messages_for_router)
        
        parsed_output = _robust_router_parser.parse(llm_response.content)
        intent = parsed_output.intent
        print(f"🧭 Router LLM chose: {intent.value} (from raw response: '{llm_response.content.strip()}')")
        return intent
    except Exception as e:
        raw_content = llm_response.content.strip() if llm_response else "N/A"
        warnings.warn(f"Router LLM failed to make a decision or parse output: {e}. Raw response: '{raw_content}'. Falling back to CHAT.")
        return Intent.CHAT

# ─── public API (used by main.py) ───────────────────────────
async def route_query(user_msg: str,
                      history: List[dict],
                      chat_id: str) -> str:
    """
    Routes the user's query to the appropriate agent based on LLM intent classification.
    """
    intent = _choose_intent(user_msg, history)
    handler = DEST[intent]
    return await handler(user_msg, history, chat_id)
