"""
src/integrations/__init__.py
────────────────────────────
Lazy factories + back-compat convenience variables.
"""

from functools import lru_cache

from .mcp_client import MCPClient
from .weaviate_client import WeaviateVectorClient
from .redis_client import get_redis, Blackboard

import logging
logging.basicConfig(
    level=logging.DEBUG,                        # DEBUG, INFO, WARNING, …
    format="%(asctime)s | %(name)12s | %(levelname)5s | %(message)s",
)

# ─── Lazy factories ──────────────────────────────────────────────
@lru_cache
def get_mcp_client() -> MCPClient:
    return MCPClient()


@lru_cache
def get_weaviate_client() -> WeaviateVectorClient:
    return WeaviateVectorClient()


@lru_cache
def get_blackboard() -> Blackboard:
    return Blackboard(get_redis())


# ─── Back-compat singletons (created on first access) ────────────
redis_conn = get_redis()        # for graph_builder & others
blackboard = get_blackboard()

__all__ = [
    "get_mcp_client",
    "get_weaviate_client",
    "get_blackboard",
    "redis_conn",
    "blackboard",
]
