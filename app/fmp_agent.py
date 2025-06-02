# app/fmp_agent.py
import json, logging, re, datetime as dt, requests
from difflib import get_close_matches
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI          # <- light LLM call
from langchain.prompts import ChatPromptTemplate

from app.config import get_settings


# ───────────────────── small utilities ─────────────────────
def _days_from_phrase(text: str) -> int | None:
    """Parse '5 days', '2 weeks', 'YTD' → int days."""
    if re.search(r"\bYTD\b", text, re.I):
        jan1 = dt.date(dt.date.today().year, 1, 1)
        return (dt.date.today() - jan1).days
    m = re.search(r"(\d+)\s*(day|week|month|year)s?", text, re.I)
    if not m:
        return None
    num, unit = int(m.group(1)), m.group(2).lower()
    return {"day": num, "week": num*7, "month": num*30, "year": num*365}[unit]


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100 if b else 0.0


def _trend(p: float) -> str:
    return "📈" if p > 0 else "📉" if p < 0 else "➖"


# ───────────────────── agent class ─────────────────────────
class FMPAgent:
    _route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a planner. Convert the user's question into JSON.\n"
             "Keys: metric (quote|fundamentals|history|earnings), "
             "lookback_days (int or null), additional (string). "
             "Return only JSON."),
            ("user", "{question}")
        ]
    )
    _llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def __init__(self, base: str, user: str, pwd: str):
        self.base = base.rstrip("/")
        self.auth = (user, pwd)

    # ------------- public entry -------------
    def run(self, query: str) -> str:
        symbol = self._symbol_from_text(query)
        if not symbol:
            return "⚠️ I couldn’t map that company name to a stock symbol."

        plan = self._plan(query)
        m = plan["metric"]
        if m == "quote":
            return self._quote(symbol)
        if m == "fundamentals":
            return self._fundamentals(symbol)
        if m == "history":
            d = plan["lookback_days"] or _days_from_phrase(query) or 30
            return self._history(symbol, d)
        if m == "earnings":
            return self._earnings(symbol)
        return "🤷 I’m not sure how to answer that."

    # ------------- planning LLM -------------
    def _plan(self, q: str) -> Dict:
        try:
            resp = self._llm.predict_messages(self._route_prompt.format(question=q))
            return json.loads(resp.content)
        except Exception:
            # fallback: heuristic
            ql = q.lower()
            if "earn" in ql:
                return {"metric": "earnings", "lookback_days": None}
            if any(w in ql for w in ("pe", "p/e", "dividend", "roe", "eps")):
                return {"metric": "fundamentals", "lookback_days": None}
            if _days_from_phrase(ql):
                return {"metric": "history", "lookback_days": _days_from_phrase(ql)}
            return {"metric": "quote", "lookback_days": None}

    # ------------- quote -------------
    def _quote(self, sym: str) -> str:
        js = self._safe("/quote", symbol=sym)
        if not js:
            return f"❌ No quote data for {sym}."
        return f"**{sym}**: ${js['price']:.2f} ({js['changesPercentage']:+.2f}% today)"

    # ------------- fundamentals -------------
    def _fundamentals(self, sym: str) -> str:
        data = self._safe("/fundamentals", symbol=sym)
        if not data:
            return f"❌ No fundamentals for {sym}."
        p, r = data["profile"], data["ratios"]
        return (
            f"**{sym} fundamentals (TTM)**\n"
            f"P/E **{r.get('priceEarningsRatioTTM','-')}** | "
            f"EPS **{r.get('epsTTM','-')}** | "
            f"ROE **{r.get('returnOnEquityTTM','-')}** | "
            f"Div Yield **{p.get('lastDiv','-')}** | "
            f"Market Cap **${p.get('mktCap',0):,}**"
        )

    # ------------- history -------------
    def _history(self, sym: str, days: int) -> str:
        js = self._safe("/historical", symbol=sym, limit=days)
        closes = [d["close"] for d in js.get("historical", [])] if js else []
        if len(closes) < 2:
            return f"❌ Not enough history for {sym}."
        pct = _pct(closes[0], closes[-1])
        return (f"**{sym} last {days} days** {_trend(pct)}\n"
                f"Change {pct:+.1f}% | Avg ${sum(closes)/len(closes):.2f} | "
                f"High ${max(closes):.2f} | Low ${min(closes):.2f}")

    # ------------- earnings -------------
    def _earnings(self, sym: str) -> str:
        js = self._safe("/historical", symbol=sym, limit=2)  # latest + prev
        hist = js.get("historical", []) if js else []
        if len(hist) < 1:
            return f"❌ No earnings data for {sym}."
        latest = hist[0]
        prev   = hist[1] if len(hist) > 1 else None
        pct = _pct(latest["close"], prev["close"]) if prev else 0.0
        return (f"**{sym} last quarter earnings** {_trend(pct)}\n"
                f"Close on report day: ${latest['close']:.2f}\n"
                f"Quarter-over-quarter change: {pct:+.1f}%")

    # ------------- symbol helpers -------------
    def _symbol_from_text(self, text: str) -> Optional[str]:
        sym = self._extract_symbol(text)
        if sym:
            return sym
        return self._lookup_symbol(text)

    @staticmethod
    def _extract_symbol(text: str) -> Optional[str]:
        m = re.search(r"\b[A-Z]{2,5}\b", text)
        return m.group(0) if m else None

    def _lookup_symbol(self, text: str) -> Optional[str]:
        q = re.sub(r"[^a-zA-Z0-9 ]", " ", text).strip()
        if len(q) < 2:
            return None
        js = self._safe("/search", query=q, limit=25)
        if not js:
            return None
        res = js.get("result", [])
        pool = {f"{r['symbol']} {r['name']}": r["symbol"] for r in res}
        best = get_close_matches(q, pool.keys(), n=1, cutoff=0.3)
        if best:
            return pool[best[0]]
        # second pass against names only
        names = {r["name"]: r["symbol"] for r in res}
        best = get_close_matches(q, names.keys(), n=1, cutoff=0.3)
        return names[best[0]] if best else None

    # ------------- HTTP helpers -------------
    def _safe(self, ep: str, **p) -> Dict[str, Any] | None:
        try:
            return self._get(ep, **p)
        except requests.HTTPError as e:
            logging.debug("FMP %s %s -> %s", ep, p, e.response.status_code)
            return None
        except Exception as e:
            logging.debug("FMP unexpected: %s", e)
            return None

    def _get(self, ep: str, **params):
        url = f"{self.base}{ep}"
        r = requests.get(url, params=params, auth=self.auth, timeout=15)
        r.raise_for_status()
        return r.json()


# factory ---------------------------------------------------------
def get_fmp_agent():
    cfg = get_settings()
    return FMPAgent(
        base_url=f"{cfg['mcp_fmp_url']}/fmp",
        user=cfg["mcp_username"],
        pwd=cfg["mcp_password"],
    )
