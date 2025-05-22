from fastapi import FastAPI, Request
from app.agent import create_agent_executor
from app.rag import get_rag_chain
from app.auth import verify_token

app = FastAPI()
agent = create_agent_executor()
rag_chain = get_rag_chain()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    token = request.headers.get("Authorization")
    user_role = verify_token(token)

    if query.startswith("context:"):
        return {"response": rag_chain.run(query[8:])}
    return {"response": await agent.arun(query, metadata={"role": user_role})}
