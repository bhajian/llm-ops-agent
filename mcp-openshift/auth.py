import os
from fastapi import HTTPException, Depends, Header

def verify_auth(authorization: str = Header(...)):
    expected_user = os.getenv("MCP_USERNAME", "admin")
    expected_pass = os.getenv("MCP_PASSWORD", "secret")
    auth_type, creds = authorization.split(" ")
    if auth_type != "Basic":
        raise HTTPException(status_code=401, detail="Invalid auth type")
    import base64
    decoded = base64.b64decode(creds).decode()
    user, pwd = decoded.split(":")
    if user != expected_user or pwd != expected_pass:
        raise HTTPException(status_code=403, detail="Invalid credentials")
