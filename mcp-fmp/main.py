# app.py ─ Parameterised FastAPI FMP Toolkit
# ─────────────────────────────────────────────────────────────
import os
import datetime as dt
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ── ENV ──────────────────────────────────────────────────────
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError(
        "FMP_API_KEY missing – set it in .env or the container environment"
    )

# ── FASTAPI APP ──────────────────────────────────────────────
app = FastAPI(
    title="FMP Toolkit",
    description="Calendars + stock data in MCP-style envelopes",
    version="2.3.0",
)

# ── FMP ENDPOINTS ────────────────────────────────────────────
BASE_V3     = "https://financialmodelingprep.com/api/v3"
BASE_STABLE = "https://financialmodelingprep.com/stable"

EP = {
    # v3
    "quote":         f"{BASE_V3}/quote",
    "analyst":       f"{BASE_V3}/analyst-stock-recommendations",
    "hist":          f"{BASE_V3}/historical-price-full",
    "grades":        f"{BASE_V3}/grade",
    "ev":            f"{BASE_V3}/enterprise-values",
    "dcf":           f"{BASE_V3}/discounted-cash-flow",
    "earn_cal":      f"{BASE_V3}/earning_calendar",
    "div_cal":       f"{BASE_V3}/stock_dividend_calendar",
    "split_cal":     f"{BASE_V3}/stock_split_calendar",
    "earn_surprise": f"{BASE_V3}/earnings-surprises",

    # stable
    "search_name":   f"{BASE_STABLE}/search-name",
}


# ── HELPERS ──────────────────────────────────────────────────
def _fetch(url: str, params: dict | None = None):
    try:
        r = requests.get(url, params=params or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, f"FMP error: {e}") from e


def _mcp(model_suffix: str, data):
    """Wrap FMP payloads into a light MCP envelope."""
    return {"model": f"financial.{model_suffix}.v1", "context": data}


def _default_dates(
    start: str | None,
    end: str | None,
    back: int = 7,
    fwd: int = 14,
) -> tuple[str, str]:
    """
    If caller omitted dates, use a sliding window around “today”.
    back / fwd are days before / after today.
    """
    today = dt.date.today()
    if not start:
        start = (today - dt.timedelta(days=back)).isoformat()
    if not end:
        end = (today + dt.timedelta(days=fwd)).isoformat()
    return start, end


# ── BODY SCHEMAS (Pydantic) ─────────────────────────────────
class SymbolBody(BaseModel):
    symbol: str = Field(..., min_length=1)


class HistBody(SymbolBody):
    limit: int | None = Field(120, ge=2, le=500)


class SurpriseBody(SymbolBody):
    limit: int | None = Field(8, ge=2, le=50)


class CalBody(SymbolBody):
    from_date: str | None = None
    to_date:   str | None = None


class DividendCalBody(CalBody):
    symbol: str | None = None


class SearchBody(BaseModel):
    query: str = Field(..., min_length=2)
    limit: int | None = Field(10, ge=1, le=50)


# ── SIMPLE AUTO-GENERATED ROUTES ────────────────────────────
_SIMPLE: list[tuple] = [
    # route name           endpoint key  schema          model suffix         qs-builder
    ("get_stock_quote",     "quote",      SymbolBody,    "quote",
     lambda b: {"apikey": FMP_API_KEY}),

    ("get_analyst_estimate","analyst",    SymbolBody,    "analyst_estimate",
     lambda b: {"apikey": FMP_API_KEY}),

    ("get_grades_summary",  "grades",     SymbolBody,    "grades_summary",
     lambda b: {"apikey": FMP_API_KEY}),

    ("get_historical_prices","hist",      HistBody,      "historical_prices",
     lambda b: {"apikey": FMP_API_KEY,
                "serietype": "line",
                "timeseries": b.limit}),

    ("get_earnings_surprises","earn_surprise", SurpriseBody, "earnings_surprises",
     lambda b: {"apikey": FMP_API_KEY, "limit": b.limit}),

    ("get_dividend_calendar","div_cal",   DividendCalBody, "dividend_calendar",
     lambda b: (
         {"apikey": FMP_API_KEY, "symbol": b.symbol}
         if b.symbol else
         {"apikey": FMP_API_KEY,
          "from": (_default_dates(b.from_date, b.to_date)[0]),
          "to":   (_default_dates(b.from_date, b.to_date)[1])}
     )),

    # NEW: search-name endpoint for ticker / company-name resolution
    ("search_symbol",       "search_name", SearchBody,  "symbol_search",
     lambda b: {"apikey": FMP_API_KEY, "query": b.query, "limit": b.limit}),
]


for route, ep_key, schema, model_suffix, qs_fn in _SIMPLE:

    async def _handler(body: schema,
                       _ep=ep_key,
                       _model=model_suffix,
                       _qs=qs_fn):
        """Universal handler factory for simple pass-through calls."""
        url = EP[_ep]

        # routes that append /{symbol}
        if hasattr(body, "symbol") and _ep not in {"div_cal", "search_name"}:
            url = f"{url}/{body.symbol}"

        return _mcp(_model, _fetch(url, _qs(body)))

    _handler.__name__ = route  # avoid FastAPI duplicate-function warning
    app.post(f"/{route}", name=route)(_handler)


# ── COMPOSITE ROUTES ───────────────────────────────────────
@app.post("/get_price_target")
async def get_price_target(body: SymbolBody):
    sym = body.symbol.upper()

    quote   = _fetch(f"{EP['quote']}/{sym}",   {"apikey": FMP_API_KEY})[0]
    analyst = _fetch(f"{EP['analyst']}/{sym}", {"apikey": FMP_API_KEY})[0]

    live, mean = quote["price"], analyst["priceTargetAverage"]
    return _mcp(
        "price_target",
        {
            "symbol":       sym,
            "price":        live,
            "mean_target":  mean,
            "upside_pct":   round((mean - live) / live * 100, 2),
            "analyst_count": analyst.get("numberOfAnalystOpinions"),
            "rating":        analyst.get("analystRating"),
        },
    )


@app.post("/get_corporate_calendar")
async def get_corporate_calendar(body: CalBody):
    sym = body.symbol.upper()
    frm, to = _default_dates(body.from_date, body.to_date)
    qs = {"apikey": FMP_API_KEY, "symbol": sym, "from": frm, "to": to}

    return _mcp(
        "corporate_calendar",
        {
            "earnings":  _fetch(EP["earn_cal"], qs),
            "splits":    _fetch(EP["split_cal"], qs),
            "dividends": _fetch(EP["div_cal"],  qs),
        },
    )


# ── ROOT ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "FMP FastAPI server",
        "docs":    "/docs",
        "tools":   [r for r, *_ in _SIMPLE] + ["get_price_target", "get_corporate_calendar"],
    }


# ── MAIN (local development) ───────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
