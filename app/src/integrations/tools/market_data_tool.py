from typing import Any, ClassVar, Optional

from langchain.tools import BaseTool

from ..mcp_client import MCPClient


class MarketDataTool(BaseTool):
    name: ClassVar[str] = "market_data"
    description: ClassVar[str] = (
        "Fetch OHLCV candles from MCP. "
        "Args: symbol(str), timeframe(str, e.g. '1h'), limit(int, ≤500). "
        "Returns JSON."
    )

    def __init__(self, client: Optional[MCPClient] = None):
        super().__init__()
        self.client = client or MCPClient()

    def _run(self, symbol: str, timeframe: str = "1h", limit: int = 100, **_: Any) -> str:
        return self.client.get_market_data(symbol, timeframe, limit).json()

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # noqa: D401
        raise NotImplementedError("Async not supported for MarketDataTool.")
