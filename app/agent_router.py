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
from typing import Callable, Dict, List, Any, Optional

from langchain.schema import SystemMessage, HumanMessage, AIMessage # Removed FunctionMessage
from pydantic import BaseModel, Field, ValidationError # Import ValidationError, for argument extraction
# FIX: Corrected import paths for output parsers (OutputFixingParser is gone)
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
# FIX: Ensure ChatPromptTemplate and MessagesPlaceholder are imported from langchain_core.prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.llm import get_llm
from app.chat_memory import chat_with_memory
# Direct imports of tools for manual invocation
from app.tools.fmp_tools import get_fmp_tools # Need access to individual FMP tool functions
from app.tools.datetime_tools import get_current_datetime_tool
from app.agents.rag_agent import get_weaviate_retriever # Still need retriever from RAG agent


# ─── dynamically load RAG helper (still useful for future expansion) ────
def _first_callable(module: str, *fn_names: str):
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        return None
    for fn_name in fn_names: # Changed from fn_names to fn_name
        obj = getattr(mod, fn_name, None)
        if callable(obj):
            return obj
    return None

# The get_rag_chain_from_module is now essentially replaced by _run_rag_agent_manually.
# Keeping the pattern here for future, but it's not directly used for invocation.
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
    # Rule 1: RAG queries (broaden to typical factual/retrieval questions)
    # These phrases typically indicate a knowledge retrieval task
    if "who is" in normalized_user_msg or \
       "what is" in normalized_user_msg or \
       "tell me about" in normalized_user_msg or \
       "information about" in normalized_user_msg or \
       "summarize" in normalized_user_msg or \
       "describe" in normalized_user_msg or \
       "explain" in normalized_user_msg: # Added more generic RAG triggers
        # Removed hardcoded names from RAG rule - the RAG agent will determine if data exists.
        print(f"🧭 Rule-based router chose: COT_RAG for query: '{user_msg}'")
        return Intent.COT_RAG

    # Rule 2: Finance queries (explicit financial terms, stock, price, company name in financial context, date/time)
    finance_keywords = ["stock", "price", "company value", "market cap", "earnings", "dividend", "financials", "analyst", "target", "eps"]
    if any(keyword in normalized_user_msg for keyword in finance_keywords) or \
       "date today" in normalized_user_msg or "current time" in normalized_user_msg or \
       "what time is it" in normalized_user_msg or "what date is it" in normalized_user_msg:
        print(f"🧭 Rule-based router chose: FINANCE for query: '{user_msg}'")
        return Intent.FINANCE
    
    # --- Fallback to LLM Router for General CHAT ---
    messages_for_router = _ROUTER_PROMPT.format_messages(
        input=user_msg,
        chat_history=_convert_history_to_messages(chat_history)
    )
    llm_response = None
    raw_content = "N/A" 
    try:
        llm_response = _router_llm.invoke(messages_for_router)
        raw_content = llm_response.content.strip() if hasattr(llm_response, 'content') else str(llm_response).strip()

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


# ─── Helper for extracting Ticker Symbol from user input ───
class TickerSymbol(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (e.g., 'NVDA' for Nvidia, 'AAPL' for Apple).")

_ticker_parser = PydanticOutputParser(pydantic_object=TickerSymbol)
_ticker_llm = get_llm(temperature=0, streaming=False) # Non-streaming for structured output

_TICKER_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert at extracting stock ticker symbols from user queries. "
        "Your task is to identify the stock symbol or company name and convert it to its ticker symbol. "
        "If the user mentions a company name, you must output its most common stock ticker symbol. "
        "If no clear company or ticker is found, output 'UNKNOWN'. "
        "You MUST output **ONLY** a JSON object with a single field 'ticker' containing the symbol. "
        "Do NOT include any conversational text, explanations, or other text outside the JSON object."
        "\n\nExample: "
        "User: 'What's the price of Apple?' "
        "Output: {{\"ticker\": \"AAPL\"}}"
        "\nUser: 'Nvidia stock?' "
        "Output: {{\"ticker\": \"NVDA\"}}"
        "\nUser: 'What about XYZ?' "
        "Output: {{\"ticker\": \"UNKNOWN\"}}"
        "\n{format_instructions}"
    )),
    HumanMessage(content="{input}"),
])

async def _extract_ticker_symbol(user_msg: str) -> Optional[str]:
    format_instructions = _ticker_parser.get_format_instructions()
    messages = _TICKER_EXTRACTION_PROMPT.format_messages(
        input=user_msg,
        format_instructions=format_instructions
    )
    llm_response = await asyncio.to_thread(_ticker_llm.invoke, messages)
    raw_response = llm_response.content.strip() if hasattr(llm_response, 'content') else str(llm_response).strip()

    try:
        # Robustly extract JSON from the LLM's response
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            json_string = match.group(0)
            parsed_ticker = _ticker_parser.parse(json_string)
            ticker = parsed_ticker.ticker.upper()
            if ticker == "UNKNOWN":
                return None
            return ticker
        else:
            warnings.warn(f"LLM did not output valid JSON for ticker extraction. Raw: '{raw_response}'")
            return None
    except (ValidationError, json.JSONDecodeError) as e:
        warnings.warn(f"Failed to parse ticker extraction JSON: {e}. Raw: '{raw_response}'")
        return None

# ─── Direct Finance Agent Logic ───────────────────────────────
async def _run_finance_direct(user_msg: str) -> str:
    normalized_user_msg = user_msg.lower().strip()
    
    # Prioritize date/time as it's a simple, direct tool call
    if "date today" in normalized_user_msg or "current time" in normalized_user_msg or \
       "what time is it" in normalized_user_msg or "what date is it" in normalized_user_msg:
        tool_output = await asyncio.to_thread(get_current_datetime_tool.func)
        return tool_output # Return directly for date/time

    # For other financial queries, extract ticker and then call FMP tools
    ticker = await _extract_ticker_symbol(user_msg)
    if not ticker:
        return "I couldn't identify the company or stock ticker you're asking about. Please specify the company name or ticker symbol."

    # Map keywords to FMP tools
    tool_func = None
    tool_name = ""
    # Get all FMP tools from the function
    fmp_tools = get_fmp_tools()
    fmp_tool_map = {t.name: t.func for t in fmp_tools}

    if "price" in normalized_user_msg or "stock quote" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_stock_quote")
        tool_name = "get_stock_quote"
    elif "analyst estimate" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_analyst_estimate")
        tool_name = "get_analyst_estimate"
    elif "price target" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_price_target")
        tool_name = "get_price_target"
    elif "historical prices" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_historical_prices")
        tool_name = "get_historical_prices"
    elif "grades summary" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_grades_summary")
        tool_name = "get_grades_summary"
    elif "corporate calendar" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_corporate_calendar")
        tool_name = "get_corporate_calendar"
    elif "dividend calendar" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_dividend_calendar")
        tool_name = "get_dividend_calendar"
    elif "earnings surprises" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_earnings_surprises")
        tool_name = "get_earnings_surprises"
    elif "enterprise valuation" in normalized_user_msg:
        tool_func = fmp_tool_map.get("get_enterprise_valuation")
        tool_name = "get_enterprise_valuation"
    elif "eps" in normalized_user_msg: # Explicitly handle EPS
        # Assuming EPS is part of get_stock_quote or a similar basic quote,
        # otherwise, you might need a dedicated FMP tool for it.
        # For now, let's try get_stock_quote and synthesize from there.
        tool_func = fmp_tool_map.get("get_stock_quote")
        tool_name = "get_stock_quote"
    else:
        return "I'm sorry, I can't determine the specific financial data you're looking for. Please be more specific (e.g., 'stock price', 'earnings per share')."

    if not tool_func:
        return "I don't have a tool to get that specific financial data."

    tool_output = ""
    try:
        # Most FMP tools take 'symbol' and potentially 'limit' or 'date'
        # For simplicity, we'll pass common arguments. This might need refinement per tool.
        # Ensure only valid args are passed, check schema if needed.
        if tool_name == "get_historical_prices":
            # Example: For historical prices, you might need to infer dates or provide defaults
            tool_output = await asyncio.to_thread(tool_func, symbol=ticker, series_type="line", from_date="2024-01-01")
        elif tool_name == "search_symbol": # This tool is used internally by LLM, not directly here.
             pass # Already extracted ticker
        else:
            tool_output = await asyncio.to_thread(tool_func, symbol=ticker)
        print(f"📊 Tool '{tool_name}' output: {tool_output}")
    except Exception as e:
        warnings.warn(f"Error calling FMP tool '{tool_name}' for {ticker}: {e}")
        return f"I encountered an error trying to get {user_msg} for {ticker}."

    # Synthesize final answer from tool output
    # This LLM call will be for final display, so set streaming=True if needed
    final_synth_llm = get_llm(temperature=0.2, streaming=True) # Set to streaming for final output

    final_synth_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful financial assistant. Based on the following user query and tool output, "
            "provide a concise and direct answer to the user. "
            "Include a concise disclaimer about the volatility of stock prices if the query was about stock data. "
            "If the tool output indicates no data or an error, state that gracefully. "
            "DO NOT include any conversational filler, preambles, or additional questions. "
            "Just the answer and the disclaimer (if applicable)."
        )),
        HumanMessage(content=user_msg),
        AIMessage(content=f"Tool Output for '{tool_name}': {tool_output}"),
    ])
    
    final_synth_response = await asyncio.to_thread(final_synth_llm.invoke, final_synth_prompt.format_messages(input=user_msg, tool_output=tool_output, tool_name=tool_name))
    return final_synth_response.content.strip() if hasattr(final_synth_response, 'content') else str(final_synth_response).strip()


async def _run_rag_agent_manually(user_msg: str, chat_history: List[dict]) -> str:
    # Get LLM and retriever
    llm = get_llm(streaming=False, temperature=0.2) # Use non-streaming LLM for sequential steps
    retriever = get_weaviate_retriever()

    # 1. Condense history to standalone question (using LLM)
    condense_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "Example: 'What is his role?' -> 'What is John Doe's role at Company X?'"
    )
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    condensed_question_messages = condense_q_prompt.format_messages(
        input=user_msg,
        chat_history=_convert_history_to_messages(chat_history)
    )
    condensed_question_llm_response = await asyncio.to_thread(llm.invoke, condensed_question_messages)
    condensed_question = condensed_question_llm_response.content.strip() if hasattr(condensed_question_llm_response, 'content') else str(condensed_question_llm_response).strip()

    print(f"🔍 RAG: Condensed question: '{condensed_question}'")

    # 2. Retrieve documents
    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, condensed_question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    if not context:
        print("🔍 RAG: No relevant documents found.")
        return "I could not find relevant information in the documents."

    print(f"🔍 RAG: Retrieved {len(retrieved_docs)} documents.")

    # 3. Answer question based on retrieved context (using LLM)
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Your goal is to answer the user's question **strictly and solely** based on the provided 'Context' documents. "
        "Synthesize information from multiple context snippets if necessary to form a complete and coherent answer. "
        "If the answer is found within the context, explicitly state that the information is from the documents. "
        "If the answer is not explicitly found within the provided context, state clearly and politely that you don't have enough information from the documents to answer. "
        "Do not use your general knowledge, make up information, infer details not explicitly present in the context, or say 'I don't know' without first confirming the information is absent from the provided context. "
        "Be concise, directly answer the question, and provide factual information only from the retrieved context. "
        "For questions about individuals (e.g., 'who is Behnam Hajian?', 'what does Matt Depue do?'), extract and present all relevant details about them from the context, such as their roles, experience, and affiliations. "
        "Do not apologize or act overly conversational; just provide the answer or state lack of information. "
        "DO NOT include ANY conversational text, preambles, questions, or explanations beyond the direct answer. Do not over-generate."
        "\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            # Removed MessagesPlaceholder(variable_name="chat_history") here
            ("human", "{input}"),
        ]
    )
    final_qa_messages = qa_prompt.format_messages(
        input=condensed_question, # Use condensed question as input
        context=context
    )

    final_answer_llm_response = await asyncio.to_thread(llm.invoke, final_qa_messages)
    return final_answer_llm_response.content.strip() if hasattr(final_answer_llm_response, 'content') else str(final_answer_llm_response).strip()


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
            # Call the new direct finance agent logic
            answer = await _run_finance_direct(user_msg) # No history needed here directly
        except Exception as e:
            warnings.warn(f"Direct Finance Agent failed: {e}")
            answer = "I'm sorry, I encountered an error while trying to get financial data."
    elif intent == Intent.COT_RAG:
        try:
            # Call the manual RAG agent logic
            answer = await _run_rag_agent_manually(user_msg, history)
        except Exception as e:
            warnings.warn(f"Manual RAG Agent failed: {e}") 
            answer = "I'm sorry, I encountered an error while trying to retrieve information from documents."
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
