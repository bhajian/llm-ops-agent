# --- app/auth.py ---
from fastapi import Header, HTTPException
import os

def verify_token(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    if token == os.getenv("ADMIN_TOKEN"):
        return "admin"
    elif token == os.getenv("USER_TOKEN"):
        return "user"
    raise HTTPException(status_code=401, detail="Invalid token")
