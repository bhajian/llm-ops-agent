# app/agents/finance_agent.py
# ────────────────────────────────────────────────────────────
"""
Utility functions for financial data processing, including
ticker extraction and answer synthesis using LLMs.
Designed to be used as nodes or helpers within the LangGraph orchestrator.
"""
from __future__ import annotations

import asyncio
import json
import re
import warnings
from typing import Any, Dict, Optional

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

# Local imports from common directory
from app.llm import get_llm


# ─── Helper for extracting Ticker Symbol from user input ───
class TickerSymbol(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (e.g., 'NVDA' for Nvidia, 'AAPL' for Apple).")

_ticker_parser = PydanticOutputParser(pydantic_object=TickerSymbol)
_ticker_llm = get_llm(temperature=0, streaming=False) # Non-streaming for structured output

_TICKER_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert at extracting stock ticker symbols from user queries. "
        "Your task is to identify the stock symbol or company name and convert it to its ticker symbol. "
        "If the user mentions a common company name for which you know the stock ticker, output that ticker. "
        "If no clear company or ticker is found, output 'UNKNOWN'. "
        "You MUST output **ONLY** a JSON object with a single field 'ticker' containing the symbol. "
        "Do NOT include any conversational text, explanations, or other text outside the JSON object."
        "\n\nExample: "
        "User: 'What's the price of Apple?' "
        "Output: {{\"ticker\": \"AAPL\"}}"
        "\nUser: 'Nvidia stock?' "
        "Output: {{\"ticker\": \"NVDA\"}}"
        "\nUser: 'What about XYZ Corp?' "
        "Output: {{\"ticker\": \"UNKNOWN\"}}"
        "\n{format_instructions}"
    )),
    HumanMessage(content="{input}"),
])

async def extract_ticker_symbol(user_msg: str) -> Optional[str]:
    """
    Uses an LLM to extract a stock ticker symbol from a user's query.
    Returns the ticker (e.g., 'AAPL') or None if not found/unknown.
    """
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

# ─── Helper for synthesizing final finance answer ───
_finance_synthesis_llm = get_llm(temperature=0.2, streaming=True) # Enable streaming for final output

async def synthesize_finance_answer(user_query: str, tool_output: Any) -> str:
    """
    Synthesizes a concise, human-readable answer for financial queries.
    Includes a disclaimer for stock data.
    """
    final_synth_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful financial assistant. Based on the following user query and tool output, "
            "provide a concise and direct answer to the user. "
            "Include a concise disclaimer about the volatility of stock prices if the query was about stock data. "
            "If the tool output indicates no data or an error, state that gracefully. "
            "DO NOT include any conversational filler, preambles, or additional questions. "
            "Just the answer and the disclaimer (if applicable)."
        )),
        HumanMessage(content=user_query),
        AIMessage(content=f"Tool Output: {tool_output}"),
    ])
    
    final_synth_response = await asyncio.to_thread(_finance_synthesis_llm.invoke, final_synth_prompt.format_messages(input=user_query, tool_output=tool_output))
    return final_synth_response.content.strip() if hasattr(final_synth_response, 'content') else str(final_synth_response).strip()

