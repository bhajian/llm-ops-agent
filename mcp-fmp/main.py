# mcp-fmp/main.py
import os, base64, secrets, httpx
from fastapi import FastAPI, Depends, Request, HTTPException, status

app = FastAPI()

# ───────── config ─────────
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE    = "https://financialmodelingprep.com/api/v3"
USER        = os.getenv("MCP_USERNAME", "admin")
PWD         = os.getenv("MCP_PASSWORD", "secret")

# ───────── auth helper ─────────
def verify(request: Request):
    expected = "Basic " + base64.b64encode(f"{USER}:{PWD}".encode()).decode()
    if not secrets.compare_digest(request.headers.get("authorization", ""), expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")

# ───────── low-level proxy ─────────
async def fmp_get(path: str, **params):
    if not FMP_API_KEY:
        raise HTTPException(500, "FMP_API_KEY not set")
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        if r.status_code == 404:
            raise HTTPException(404, f"FMP endpoint 404: {path}")
        r.raise_for_status()
        return r.json()

# ───────── endpoints ─────────
@app.get("/fmp/quote")
async def quote(symbol: str, _=Depends(verify)):
    data = await fmp_get(f"quote/{symbol.upper()}")
    return data[0] if data else {}

@app.get("/fmp/historical")
async def historical(symbol: str, limit: int = 100, _=Depends(verify)):
    return await fmp_get(
        f"historical-price-full/{symbol.upper()}",
        timeseries=limit,
        serietype="line",
    )

@app.get("/fmp/fundamentals")
async def fundamentals(symbol: str, _=Depends(verify)):
    sym = symbol.upper()
    profile, ratios = await asyncio.gather(
        fmp_get(f"profile/{sym}"),
        fmp_get(f"ratios-ttm/{sym}")
    )
    return {
        "profile": profile[0] if profile else {},
        "ratios":  ratios[0]  if ratios  else {},
    }

@app.get("/fmp/search")
async def search(query: str, limit: int = 10, _=Depends(verify)):
    """
    Company-name & ticker search used by the LLM router to resolve symbols.
    """
    return {"result": await fmp_get("search", query=query, limit=limit)}
