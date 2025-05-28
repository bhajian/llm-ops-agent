# --- mcp-server/main.py ---
from fastapi import FastAPI, Depends, Request
from openshift_client import scale_deployment, list_pods
from auth import verify_auth

app = FastAPI()


@app.post("/openshift/scale")
async def scale(request: Request, auth=Depends(verify_auth)):
    body = await request.json()                         # ➜ await
    scale_deployment(
        namespace=body["namespace"],
        name=body["deployment"],
        replicas=body["replicas"],
    )
    return {"status": "scaled"}


@app.get("/openshift/pods")
async def get_pods(namespace: str, auth=Depends(verify_auth)):
    return {"pods": list_pods(namespace)}
