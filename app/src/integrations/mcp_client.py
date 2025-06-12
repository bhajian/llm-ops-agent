"""Thin synchronous wrapper around your MCP server."""

import httpx
from pydantic import BaseModel, Field

from ..config import get_settings

_cfg = get_settings()


class MarketDataResponse(BaseModel):
    symbol: str
    timeframe: str
    candles: list[list[float]] = Field(
        description="[timestamp, open, high, low, close, volume]"
    )


class MCPClient:
    """
    Minimal MCP REST client.

    Does **not** raise at __init__; only raises when you actually call
    .get_market_data() and the base URL is missing.
    """

    def __init__(self):
        self.base_url = _cfg.mcp_base_url.rstrip("/") if _cfg.mcp_base_url else None
        self.api_key = _cfg.mcp_api_key or ""
        self._client = None  # defer httpx.Client creation

    # ------------------ public methods ------------------ #
    def get_market_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> MarketDataResponse:
        if not self.base_url:
            raise RuntimeError("MCP_BASE_URL not set")

        if self._client is None:
            self._client = httpx.Client(
                timeout=10.0, headers={"Authorization": self.api_key}
            )

        url = f"{self.base_url}/market_data"
        params = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return MarketDataResponse(**resp.json())
