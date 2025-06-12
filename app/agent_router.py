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
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
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
# You'll also need to ensure app.agents.k8s_agent is imported if used directly here,
# but currently, it seems only FMP and RAG are pulled into agent_router directly.


# ─── intent labels and Pydantic for routing output ──────────
class Intent(str, Enum):
    CHAT = "CHAT"
    FINANCE = "FINANCE"
    COT_RAG = "COT_RAG" # CoT = Chain of Thought (for RAG)


class RouterOutput(BaseModel):
    intent: Intent = Field(
        ...,
        description=(
            "The intent of the user's message. "
            "Choose 'FINANCE' if the user is asking a question about stocks, company financials, or any financial data. "
            "Choose 'COT_RAG' if the user is asking a question that can be answered by retrieving information from documents. "
            "Otherwise, choose 'CHAT' for general conversation or if no specific intent is detected."
        )
    )

_router_output_schema_json_string = json.dumps(RouterOutput.model_json_schema(), indent=2)


_router_llm = get_llm(temperature=0)
_robust_router_parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=RouterOutput), llm=_router_llm)


_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert at routing user questions or statements to the most appropriate agent or action "
        "based on the user's intent. "
        "Your decision should be based *solely* on the current user input and the preceding chat history. "
        "You MUST choose one of the following intents and provide ONLY the JSON output, with no additional text or explanation."
        "\n\n--- Available Intents ---"
        "\n- CHAT: Select 'CHAT' for general conversation, greetings, personal questions (e.g., 'What is my name?'), "
        "or statements providing information about the user (e.g., 'My name is Behnam', 'I like pizza'). "
        "This is the **default and broadest intent** if no other specific intent clearly applies. "
        "Choose CHAT if the query lacks clear financial keywords or a request for document-based information, or if it's a simple conversational turn."
        "\n- FINANCE: **CRITICAL**: Choose 'FINANCE' ONLY if the user is asking specifically about **stock prices, company financials, earnings, market news, dividends, price targets, or any other data directly related to financial markets for a specific company or the market as a whole.** "
        "This intent *requires* the use of specialized financial tools. Do NOT choose FINANCE for general conversation, personal information, or non-financial topics."
        "\n- COT_RAG: Select 'COT_RAG' ONLY for questions that require retrieving factual information from internal documents, knowledge bases, or asking for details from previously provided text. "
        "This includes queries about people, events, concepts, or summaries found within your indexed documents. "
        "Examples: 'Who is Bahram Hajian?', 'What is the history of AI?', 'Summarize the document about project X.', 'What are the key findings of the report?'"
        "\n\n--- Example of Intent Routing ---"
        "\nUser Input: 'Hello there!' -> Intent: CHAT"
        "\nUser Input: 'What is Apple's stock price?' -> Intent: FINANCE"
        "\nUser Input: 'Summarize the document I just uploaded.' -> Intent: COT_RAG"
        "\nUser Input: 'My favorite color is blue.' -> Intent: CHAT"
        "\nUser Input: 'Tell me about the new security policy.' -> Intent: COT_RAG" # Explicit example for RAG
        "\n\n{format_instructions}"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
])


def _choose_intent(user_msg: str, chat_history: List[dict]) -> Intent:
    format_instructions = _robust_router_parser.get_format_instructions()
    
    messages_for_router = _ROUTER_PROMPT.format_messages(
        input=user_msg,
        chat_history=_convert_history_to_messages(chat_history),
        format_instructions=format_instructions
    )
    llm_response = None
    raw_content = "N/A" # Initialize raw_content outside try block
    try:
        llm_response = _router_llm.invoke(messages_for_router)
        
        # Safely extract content from llm_response
        if hasattr(llm_response, 'content'):
            raw_content = llm_response.content.strip()
        else:
            raw_content = str(llm_response).strip()

        parsed_output = _robust_router_parser.parse(raw_content)
        intent = parsed_output.intent
        print(f"🧭 Router LLM chose: {intent.value} (from raw response: '{raw_content}')")
        return intent
    except Exception as e:
        warnings.warn(f"Router LLM failed to make a decision or parse output: {e}. Raw response: '{raw_content}'. Falling back to CHAT.")
        return Intent.CHAT

# ─── public API (used by main.py) ───────────────────────────
async def route_query(user_msg: str,
                      history: List[dict],
                      chat_id: str) -> str:
    """
    Routes the user's query to the appropriate agent based on LLM decision.
    """
    # 1. Choose intent via LLM
    intent = _choose_intent(user_msg, history)

    # 2. Call appropriate agent based on intent
    answer = ""
    if intent == Intent.CHAT:
        answer = chat_with_memory(user_msg, history)
    elif intent == Intent.FINANCE:
        # TODO: Ensure FMP agent uses async `ainvoke` if available for performance
        try:
            fmp_agent = get_fmp_agent()
            # If the agent is async, use ainvoke, otherwise invoke in a thread
            if hasattr(fmp_agent, "ainvoke"):
                result = await fmp_agent.ainvoke({"input": user_msg, "chat_history": _convert_history_to_messages(history)})
            else:
                result = await asyncio.to_thread(fmp_agent.invoke, {"input": user_msg, "chat_history": _convert_history_to_messages(history)})
            answer = result.get("output", "Could not get an answer from the FMP agent.")
        except Exception as e:
            warnings.warn(f"FMP Agent failed during ainvoke: {e}")
            answer = "I'm sorry, I encountered an error while trying to get financial data."
    elif intent == Intent.COT_RAG:
        if get_rag_chain_from_module:
            try:
                rag_chain = get_rag_chain_from_module()
                if hasattr(rag_chain, "ainvoke"):
                    result = await rag_chain.ainvoke({"input": user_msg, "chat_history": _convert_history_to_messages(history)})
                else:
                    result = await asyncio.to_thread(rag_chain.invoke, {"input": user_msg, "chat_history": _convert_history_to_messages(history)})
                answer = result.get("answer", "I could not find relevant information in the documents.")
            except Exception as e:
                warnings.warn(f"RAG Agent failed during ainvoke: {e}")
                answer = "I'm sorry, I encountered an error while trying to retrieve information from documents."
        else:
            warnings.warn("RAG agent module 'app.agents.rag_agent' or 'get_rag_chain' not found. Falling back to CHAT.")
            answer = chat_with_memory(user_msg, history) # Fallback to chat if RAG is not available
    else:
        # Fallback for any unhandled intent, though enum should prevent this
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
