# mcp-fmp/main.py
from fastapi import FastAPI, Depends, Request, HTTPException, status
import httpx, os, base64, secrets

app = FastAPI()

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE    = "https://financialmodelingprep.com/api/v3"

# ───────── auth helper ─────────
def verify(request: Request):
    good = "Basic " + base64.b64encode(
        f"{os.getenv('MCP_USERNAME','admin')}:{os.getenv('MCP_PASSWORD','secret')}".encode()
    ).decode()
    if not secrets.compare_digest(request.headers.get("authorization", ""), good):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")

# ───────── low-level proxy ─────────
async def fmp_get(path: str, **params):
    if not FMP_API_KEY:
        raise HTTPException(500, "FMP_API_KEY not set")
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

# ───────── endpoints ─────────
@app.get("/fmp/quote")
async def quote(symbol: str, _=Depends(verify)):
    data = await fmp_get(f"quote/{symbol.upper()}")
    return data[0] if data else {}

@app.get("/fmp/historical")
async def historical(symbol: str, limit: int = 100, _=Depends(verify)):
    return await fmp_get(f"historical-price-full/{symbol.upper()}",
                         timeseries=limit, serietype="line")

@app.get("/fmp/fundamentals")
async def fundamentals(symbol: str, _=Depends(verify)):
    sym = symbol.upper()
    profile  = (await fmp_get(f"profile/{sym}"))[0] if await fmp_get(f"profile/{sym}") else {}
    ratios   = (await fmp_get(f"ratios-ttm/{sym}"))[0] if await fmp_get(f"ratios-ttm/{sym}") else {}
    return {"profile": profile, "ratios": ratios}
