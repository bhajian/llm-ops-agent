# app/fmp_agent.py
import re, requests
from typing import Dict, Any

from app.config import get_settings


class FMPAgent:
    """
    Facade over the MCP-FMP micro-service.
    """

    def __init__(self, base_url: str, user: str, pwd: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth     = (user, pwd)         # HTTP Basic creds

    # ───────── public entry ─────────
    def run(self, query: str) -> str:
        symbol = self._extract_symbol(query)
        if symbol is None:
            return "⚠️ I couldn’t detect a stock symbol in your question."

        q = query.lower()

        if any(w in q for w in ("pe", "p/e", "ratio", "fundamental", "dividend", "market cap", "roe", "eps")):
            return self._fundamentals(symbol)

        if any(w in q for w in ("history", "historical", "chart", "past")):
            return self._history(symbol)

        return self._quote(symbol)

    # ───────── helpers ─────────
    def _quote(self, sym: str) -> str:
        js = self._get("/quote", symbol=sym)
        if not js:
            return f"❌ No quote for {sym}."
        q = js
        return f"**{sym}**: ${q['price']:.2f}  ({q['changesPercentage']:+.2f}% today)"

    def _history(self, sym: str) -> str:
        js = self._get("/historical", symbol=sym, limit=30)
        prices = js.get("historical", [])
        if not prices:
            return f"❌ No historical data for {sym}."
        close, old = prices[0]["close"], prices[-1]["close"]
        pct = (close - old) / old * 100
        return f"{sym} closed at **${close:.2f}** yesterday; 30-day change **{pct:+.1f}%**."

    def _fundamentals(self, sym: str) -> str:
        js = self._get("/fundamentals", symbol=sym)
        prof, rat = js["profile"], js["ratios"]
        if not prof:
            return f"❌ No fundamentals for {sym}."
        parts = [
            f"**{sym} fundamentals (TTM)**",
            f"P/E **{rat.get('priceEarningsRatioTTM','-')}**",
            f"EPS **{rat.get('epsTTM','-')}**",
            f"ROE **{rat.get('returnOnEquityTTM','-')}**",
            f"Div Yield **{prof.get('lastDiv','-')}**",
            f"Market Cap **${prof.get('mktCap',0):,}**",
        ]
        return " | ".join(parts)

    # REST proxy call
    def _get(self, endpoint: str, **params) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        r = requests.get(url, params=params, auth=self.auth, timeout=10)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _extract_symbol(text: str) -> str | None:
        m = re.search(r"\b[A-Z]{1,5}\b", text)
        return m.group(0) if m else None


# ───────── factory ──────────
def get_fmp_agent() -> FMPAgent:
    cfg = get_settings()
    return FMPAgent(
        base_url=f"{cfg['mcp_fmp_url']}/fmp",
        user=cfg["mcp_username"],
        pwd=cfg["mcp_password"],
    )
