from fastapi import FastAPI, Depends, Request, HTTPException, status
import httpx, os, base64, secrets

app = FastAPI()
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE    = "https://financialmodelingprep.com/api/v3"

# ---------------- auth helper ----------------
def verify(request: Request):
    auth = request.headers.get("authorization", "")
    target = "Basic " + base64.b64encode(
        f"{os.getenv('MCP_USERNAME','admin')}:{os.getenv('MCP_PASSWORD','secret')}".encode()
    ).decode()
    if not secrets.compare_digest(auth, target):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad credentials")

# ---------------- tiny FMP proxy ----------------
async def fmp_get(endpoint: str, **params):
    params["apikey"] = FMP_API_KEY
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{FMP_BASE}/{endpoint}", params=params)
        r.raise_for_status()
        return r.json()

@app.get("/fmp/quote")
async def quote(symbol: str, _=Depends(verify)):
    data = await fmp_get("quote", symbol=symbol.upper())
    return data[0] if data else {}

@app.get("/fmp/historical")
async def historical(symbol: str, limit: int = 100, _=Depends(verify)):
    return await fmp_get(f"historical-price-full/{symbol.upper()}", timeseries=limit, serietype="line")
