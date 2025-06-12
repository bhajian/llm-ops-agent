
src/
  integrations/             # ⇦ connectors + tool wrappers live here
    __init__.py
    mcp_client.py           # raw MCP HTTP/client helpers
    weaviate_client.py      # thin Weaviate SDK wrapper
    redis_client.py         # Redis connection factory (both KV & Vector)
    

    tools/                  # LangChain‑compatible wrappers built on the above
      market_data_tool.py   # uses MCP client
      vector_search_tool.py # uses Weaviate or RedisVector
      memory_tool.py        # wraps RedisConversationMemory / Blackboard

  agents/
    finance_agent.py
    retrieval_agent.py
    reasoner_agent.py
    k8s_agent.py

  graph/
    graph_builder.py        # builds LangGraph DAG

  api/
    server.py               # FastAPI app, SSE/WebSocket streaming

  config.py                 # env / settings loader
  llm.py                    # Bedrock / OpenAI / vLLM factory
  __init__.py
pyproject.toml
README.md
.env