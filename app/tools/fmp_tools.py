"""
app/tools/fmp_tools.py
─────────────────────────────────────────────────────────────
Light LangChain wrapper around the MCP-FMP FastAPI server.

Every function POSTs to the service, unwraps the MCP envelope, and is
exposed to an agent as a LangChain StructuredTool.
"""
from __future__ import annotations

import os
from typing import Any, List, Dict

import httpx
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


# ───────────────────────── config ──────────────────────────
# Docker-compose: service name is “mcp-fmp”.  Allow override for local dev.
FMP_BASE = os.getenv("FMP_FASTAPI_URL", "http://mcp-fmp:5000")

# fallback headers (httpx uses none by default)
_HEADERS: Dict[str, str] = {"Content-Type": "application/json"}


# ───────────────────────── helper ──────────────────────────
def _post(path: str, payload: dict) -> Any:
    """
    Thin wrapper:  POST  →  {FMP_BASE}{path}
    · Raises on any HTTP error.
    · Returns the inner “context” part of the MCP envelope.
    """
    r = httpx.post(f"{FMP_BASE}{path}", json=payload, headers=_HEADERS, timeout=15)
    r.raise_for_status()
    body = r.json()
    # basic guard – all endpoints wrap data the same way
    if not isinstance(body, dict) or "context" not in body:
        raise ValueError(f"Unexpected MCP response shape from {path}: {body}")
    return body["context"]


# ───────────────────────── tool args schemas ───────────────
class _SymbolArgs(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol (e.g. NVDA, AAPL)")

class _SymbolLimitArgs(_SymbolArgs):
    limit: int = Field(1, description="Number of results to return (default: 1)")

class _SymbolDateArgs(_SymbolArgs):
    date: str = Field(..., description="Date in YYYY-MM-DD format")

class _SearchArgs(BaseModel):
    query: str = Field(..., description="Company name or partial ticker to search for")
    limit: int = Field(5, description="Number of results to return (default: 5)")


# ───────────────────────── FMP tool functions ──────────────
def _get_stock_quote(symbol: str) -> dict:
    """
    Returns latest stock quote for a given ticker symbol.
    Provides fields like: price, volume, marketCap, eps, etc.
    """
    return _post("/quote", {"symbol": symbol})


def _get_analyst_estimate(symbol: str, limit: int = 1) -> List[dict]:
    """
    Returns analyst estimates for a given ticker symbol.
    Set limit=1 for the latest.
    """
    return _post("/analyst_estimates", {"symbol": symbol, "limit": limit})


def _get_price_target(symbol: str, limit: int = 1) -> List[dict]:
    """
    Returns analyst price targets for a given ticker symbol.
    Set limit=1 for the latest.
    """
    return _post("/price_target", {"symbol": symbol, "limit": limit})


def _get_historical_prices(
    symbol: str, from_date: str, to_date: str = "", series_type: str = "line"
) -> List[dict]:
    """
    Returns historical daily prices for a given ticker symbol.
    `from_date` and `to_date` are YYYY-MM-DD.
    `series_type` can be "line" (default), "bar", "candlestick".
    """
    return _post(
        "/historical_prices",
        {"symbol": symbol, "from": from_date, "to": to_date, "serietype": series_type},
    )


def _get_grades_summary(symbol: str, limit: int = 1) -> List[dict]:
    """
    Returns a summary of analyst recommendations (grades) for a given ticker symbol.
    """
    return _post("/grade", {"symbol": symbol, "limit": limit})


def _get_corporate_calendar(from_date: str, to_date: str = "") -> List[dict]:
    """
    Returns corporate calendar events like earnings, IPOs, dividends.
    `from_date` and `to_date` are YYYY-MM-DD.
    """
    return _post("/corporate_calendar", {"from": from_date, "to": to_date})


def _get_dividend_calendar(from_date: str, to_date: str = "") -> List[dict]:
    """
    Returns dividend calendar events.
    `from_date` and `to_date` are YYYY-MM-DD.
    """
    return _post("/dividend_calendar", {"from": from_date, "to": to_date})


def _get_earnings_surprises(symbol: str, limit: int = 1) -> List[dict]:
    """
    Returns actual vs. estimated EPS (Earnings Per Share) surprises for a symbol.
    """
    return _post("/earnings_surprises", {"symbol": symbol, "limit": limit})


def _get_enterprise_valuation(symbol: str, limit: int = 1) -> List[dict]:
    """
    Returns enterprise valuation metrics (e.g., marketCap, enterpriseValue) for a symbol.
    """
    return _post("/enterprise_valuation", {"symbol": symbol, "limit": limit})


def _search_symbol(query: str, limit: int = 5) -> List[dict]:
    """
    Resolve company names → ticker symbols using the MCP server’s
    `/search_symbol` route (backed by FMP “stable/search-name”).
    """
    return _post("/search_symbol", {"query": query, "limit": limit})


# ───────────────────────── tool registry ───────────────────
# Note: In the LangGraph redesign, these StructuredTool objects are primarily
# used for their descriptions and schemas, which can be dynamically used by the LLM.
# The underlying functions are called directly by the orchestrator nodes.
TOOLS: List[StructuredTool] = [
    StructuredTool.from_function(_get_stock_quote,        name="get_stock_quote", args_schema=_SymbolArgs),
    StructuredTool.from_function(_get_analyst_estimate,   name="get_analyst_estimate", args_schema=_SymbolLimitArgs),
    StructuredTool.from_function(_get_price_target,       name="get_price_target", args_schema=_SymbolLimitArgs),
    StructuredTool.from_function(_get_historical_prices,  name="get_historical_prices", args_schema=_SymbolDateArgs),
    StructuredTool.from_function(_get_grades_summary,     name="get_grades_summary", args_schema=_SymbolLimitArgs),
    StructuredTool.from_function(_get_corporate_calendar, name="get_corporate_calendar", args_schema=_SymbolDateArgs),
    StructuredTool.from_function(_get_dividend_calendar,  name="get_dividend_calendar", args_schema=_SymbolDateArgs),
    StructuredTool.from_function(_get_earnings_surprises, name="get_earnings_surprises", args_schema=_SymbolLimitArgs),
    StructuredTool.from_function(_get_enterprise_valuation, name="get_enterprise_valuation", args_schema=_SymbolLimitArgs),
    StructuredTool.from_function(_search_symbol,          name="search_symbol", args_schema=_SearchArgs),
]


def get_fmp_tools() -> List[StructuredTool]:
    """Convenience accessor consumed by orchestrator for tool descriptions."""
    return TOOLS
