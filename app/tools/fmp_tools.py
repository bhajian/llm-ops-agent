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


# ───────────────────────── single-arg tools ────────────────
def _get_stock_quote(symbol: str):
    """Live price, change %, market-cap, etc."""
    return _post("/get_stock_quote", {"symbol": symbol})


def _get_analyst_estimate(symbol: str):
    """Mean analyst target & rating."""
    return _post("/get_analyst_estimate", {"symbol": symbol})


def _get_price_target(symbol: str):
    """Current price vs mean target and upside %."""
    return _post("/get_price_target", {"symbol": symbol})


def _get_historical_prices(symbol: str, limit: int = 120):
    """Daily OHLCV (max 500 trailing sessions)."""
    limit = max(2, min(limit, 500))
    return _post("/get_historical_prices", {"symbol": symbol, "limit": limit})


def _get_grades_summary(symbol: str):
    """Overall / growth / profitability / valuation grades."""
    return _post("/get_grades_summary", {"symbol": symbol})


def _get_corporate_calendar(symbol: str):
    """Upcoming earnings, splits, dividends (±2 weeks)."""
    return _post("/get_corporate_calendar", {"symbol": symbol})


def _get_dividend_calendar(symbol: str):
    """Dividend events for this ticker (±2 weeks)."""
    return _post("/get_dividend_calendar", {"symbol": symbol})


def _get_earnings_surprises(symbol: str, limit: int = 8):
    """Last *n* EPS surprises (2 ≤ n ≤ 50)."""
    limit = max(2, min(limit, 50))
    return _post("/get_earnings_surprises", {"symbol": symbol, "limit": limit})


def _get_enterprise_valuation(symbol: str):
    """
    Enterprise-value ratios + last discounted-cash-flow calc.
    (Server must expose /get_enterprise_valuation.)
    """
    return _post("/get_enterprise_valuation", {"symbol": symbol})


# ───────────────────────── NEW search tool ─────────────────
class _SearchArgs(BaseModel):
    query: str = Field(..., description="Company name or partial ticker")
    limit: int = Field(10, ge=1, le=50, description="Max matches to return")


def _search_symbol(query: str, limit: int = 10):
    """
    Resolve company names → ticker symbols using the MCP server’s
    `/search_symbol` route (backed by FMP “stable/search-name”).
    """
    return _post("/search_symbol", {"query": query, "limit": limit})


# ───────────────────────── tool registry ───────────────────
TOOLS: List[StructuredTool] = [
    StructuredTool.from_function(_get_stock_quote,        name="get_stock_quote"),
    StructuredTool.from_function(_get_analyst_estimate,   name="get_analyst_estimate"),
    StructuredTool.from_function(_get_price_target,       name="get_price_target"),
    StructuredTool.from_function(_get_historical_prices,  name="get_historical_prices"),
    StructuredTool.from_function(_get_grades_summary,     name="get_grades_summary"),
    StructuredTool.from_function(_get_corporate_calendar, name="get_corporate_calendar"),
    StructuredTool.from_function(_get_dividend_calendar,  name="get_dividend_calendar"),
    StructuredTool.from_function(_get_earnings_surprises, name="get_earnings_surprises"),
    StructuredTool.from_function(_get_enterprise_valuation,
                                 name="get_enterprise_valuation"),
    # NEW: ticker lookup by name
    StructuredTool.from_function(
        _search_symbol,
        name="search_symbol",
        args_schema=_SearchArgs,
        description="Lookup ticker symbols by company name or partial ticker",
    ),
]


def get_fmp_tools() -> List[StructuredTool]:
    """Convenience accessor consumed by FMPAgent or other chains."""
    return TOOLS
