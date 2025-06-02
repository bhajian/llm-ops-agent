# app/fmp_agent.py
from __future__ import annotations

import json, logging, re, datetime as dt, requests
from difflib import get_close_matches
from typing   import Dict, Any, Optional

from langchain_openai import ChatOpenAI            # lightweight LLM call
from langchain.prompts import ChatPromptTemplate

from app.config import get_settings


# ───────────────────────── small utilities ───────────────────────────────
def _days_from_phrase(text: str) -> int | None:
    """Parse phrases like '5 days', '3 months', 'YTD' → number of days."""
    if re.search(r"\bYTD\b", text, re.I):
        jan1 = dt.date(dt.date.today().year, 1, 1)
        return (dt.date.today() - jan1).days

    m = re.search(r"(\d+)\s*(day|week|month|year)s?", text, re.I)
    if not m:
        return None
    num, unit = int(m.group(1)), m.group(2).lower()
    return {"day": num,
            "week": num * 7,
            "month": num * 30,
            "year": num * 365}[unit]


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100 if b else 0.0


def _trend(p: float) -> str:
    return "📈" if p > 0 else "📉" if p < 0 else "➖"


# ─────────────────────────── agent class ─────────────────────────────────
class FMPAgent:
    """
    Facade over the MCP-FMP micro-control-plane.
    Decides what data to fetch (quote, fundamentals, history, earnings)
    via a fast chat-prompt planner.
    """

    _route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a planner. Convert the user question into JSON.\n"
             "Keys: metric (quote|fundamentals|history|earnings), "
             "lookback_days (int or null), additional (string).\n"
             "Return only JSON."),
            ("user", "{question}"),
        ]
    )
    _llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def __init__(self, base: str, user: str, pwd: str) -> None:
        self.base = base.rstrip("/")
        self.auth = (user, pwd)

    # ────────────── public entry ────────────────────────────────────────
    def run(self, query: str) -> str:
        symbol = self._symbol_from_text(query)
        if not symbol:
            return "⚠️ I couldn’t map that company name to a stock symbol."

        plan   = self._plan(query)
        metric = plan["metric"]

        if metric == "quote":
            return self._quote(symbol)
        if metric == "fundamentals":
            return self._fundamentals(symbol)
        if metric == "history":
            lookback = (plan["lookback_days"]
                        or _days_from_phrase(query)
                        or 30)
            return self._history(symbol, lookback)
        if metric == "earnings":
            return self._earnings(symbol)

        return "🤷 I’m not sure how to answer that."

    # ────────────── planning LLM ────────────────────────────────────────
    def _plan(self, q: str) -> Dict:
        try:
            resp = self._llm.predict_messages(
                self._route_prompt.format(question=q)
            )
            return json.loads(resp.content)
        except Exception:
            # heuristic fallback
            ql = q.lower()
            if "earn" in ql:
                return {"metric": "earnings", "lookback_days": None}
            if any(w in ql for w in ("pe", "p/e", "dividend", "roe", "eps")):
                return {"metric": "fundamentals", "lookback_days": None}
            if _days_from_phrase(ql):
                return {"metric": "history",
                        "lookback_days": _days_from_phrase(ql)}
            return {"metric": "quote", "lookback_days": None}

    # ────────────── quote ---------------------------------------------------------------
    def _quote(self, sym: str) -> str:
        js = self._safe("/quote", symbol=sym)
        if not js:
            return f"❌ No quote data for {sym}."
        return f"**{sym}**: ${js['price']:.2f} ({js['changesPercentage']:+.2f}% today)"

    # ────────────── fundamentals --------------------------------------------------------
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

    # ────────────── history -------------------------------------------------------------
    def _history(self, sym: str, days: int) -> str:
        js = self._safe("/historical", symbol=sym, limit=days)
        closes = [d["close"] for d in js.get("historical", [])] if js else []
        if len(closes) < 2:
            return f"❌ Not enough history for {sym}."
        pct = _pct(closes[0], closes[-1])
        return (
            f"**{sym} last {days} days** {_trend(pct)}\n"
            f"Change {pct:+.1f}% | "
            f"Avg ${sum(closes)/len(closes):.2f} | "
            f"High ${max(closes):.2f} | "
            f"Low ${min(closes):.2f}"
        )

    # ────────────── earnings ------------------------------------------------------------
    def _earnings(self, sym: str) -> str:
        js = self._safe("/historical", symbol=sym, limit=2)  # latest + prev
        hist = js.get("historical", []) if js else []
        if not hist:
            return f"❌ No earnings data for {sym}."
        latest = hist[0]
        prev   = hist[1] if len(hist) > 1 else None
        pct    = _pct(latest["close"], prev["close"]) if prev else 0.0
        return (
            f"**{sym} last-quarter earnings** {_trend(pct)}\n"
            f"Close on report day: ${latest['close']:.2f}\n"
            f"QoQ change: {pct:+.1f}%"
        )

    # ────────────── symbol helpers ------------------------------------------------------
    def _symbol_from_text(self, text: str) -> Optional[str]:
        return self._extract_symbol(text) or self._lookup_symbol(text)

    @staticmethod
    def _extract_symbol(text: str) -> Optional[str]:
        m = re.search(r"\b[A-Z]{2,5}\b", text)
        return m.group(0) if m else None

    def _lookup_symbol(self, text: str) -> Optional[str]:
        """
        Resolve *company name* → ticker.

        Strategy
        --------
        1.  Send the whole (cleaned) query to `/search`.
            • If we get hits → keep them.
        2.  If no hits, retry *each* meaningful token (`Nvidia`, `Apple`, …).
            • First token that yields results wins.
        3.  Apply substring-then-fuzzy matching on the collected hits.
        """
        # --- 0) build list of candidate queries ---------------------------
        q_raw   = re.sub(r"[^a-zA-Z0-9 ]", " ", text).strip()
        tokens  = [t for t in q_raw.split() if len(t) > 2]          # cheap stop-word filter
        queries = [q_raw] + tokens                                  # try full sentence first

        results: list[dict] = []
        for q in queries:
            js = self._safe("/search", query=q, limit=25)
            hits = js.get("result", []) if js else []
            if hits:
                results = hits
                break                                               # stop at first success

        if not results:                                             # nothing at all
            return None

        q_lo = q_raw.lower()

        # --- 1) direct substring match -----------------------------------
        for rec in results:
            if rec["symbol"].lower() in q_lo:
                return rec["symbol"]
            # allow individual words of the company-name to match
            if any(word in q_lo for word in rec["name"].lower().split()):
                return rec["symbol"]

        # --- 2) fuzzy fallback (unchanged) -------------------------------
        pool = {f"{r['symbol']} {r['name']}": r["symbol"] for r in results}
        best = get_close_matches(q_lo, (k.lower() for k in pool.keys()), n=1, cutoff=0.3)
        if best:
            match_key = next(k for k in pool if k.lower() == best[0])
            return pool[match_key]

        names = {r["name"]: r["symbol"] for r in results}
        best  = get_close_matches(q_lo, (n.lower() for n in names), n=1, cutoff=0.3)
        if best:
            name_key = next(n for n in names if n.lower() == best[0])
            return names[name_key]

        return None

    # ────────────── HTTP helpers --------------------------------------------------------
    def _safe(self, ep: str, **p) -> Dict[str, Any] | None:
        try:
            return self._get(ep, **p)
        except requests.HTTPError as e:
            logging.debug("FMP %s %s → %s", ep, p, e.response.status_code)
            return None
        except Exception as e:
            logging.debug("FMP unexpected: %s", e)
            return None

    def _get(self, ep: str, **params):
        url = f"{self.base}{ep}"
        r   = requests.get(url, params=params, auth=self.auth, timeout=15)
        r.raise_for_status()
        return r.json()


# ───────────────────────── singleton factory ─────────────────────────────
_fmp_singleton: FMPAgent | None = None


def get_fmp_agent() -> FMPAgent:
    """Return a single shared FMPAgent instance."""
    global _fmp_singleton
    if _fmp_singleton is None:
        cfg = get_settings()
        _fmp_singleton = FMPAgent(
            base=f"{cfg['mcp_fmp_url'].rstrip('/')}/fmp",
            user=cfg["mcp_username"],
            pwd =cfg["mcp_password"],
        )
    return _fmp_singleton
