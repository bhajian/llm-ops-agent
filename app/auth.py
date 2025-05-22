import os
def verify_token(token: str):
    if token == f"Bearer {os.environ['ADMIN_TOKEN']}":
        return "admin"
    raise Exception("Unauthorized")
