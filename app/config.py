# app/config.py
"""
Single source of truth for runtime settings.

▪ Works out-of-the-box with OpenAI cloud
▪ Switch to vLLM / llama by exporting:
     LLM_BACKEND=vllm
     LLM_BASE_URL=http://vllm:8000/v1
     LLM_MODEL=llama-3-8b-instruct
"""

import os
from functools import lru_cache
from typing import Dict, Any


def _env(key: str, default: str | None = None) -> str | None:
    """Tiny helper for env reads (keeps typing happy)."""
    return os.getenv(key, default)


@lru_cache
def get_settings() -> dict:
    return {
        # ─── Bedrock ──────────────────────────────────────────
        "bedrock_region":  _env("BEDROCK_REGION", _env("AWS_REGION", "us-east-2")),
        "bedrock_profile": _env("AWS_PROFILE"),          # optional named credentials
        "bedrock_embedding_model": _env("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"), # <--- ADDED/FIXED THIS LINE
        # ─── Data Layer ─────────────────────────────────────
        "redis_host":    _env("REDIS_HOST", "redis"),
        "weaviate_url":  _env("WEAVIATE_URL", "http://weaviate:8080"),

        # ─── MCP Proxies (FMP, K8s/OC) ─────────────────────
        "mcp_fmp_url":   _env("MCP_FMP_URL",  "http://mcp-fmp:9000"),
        "mcp_k8s_url":   _env("MCP_K8S_URL",  "http://mcp-openshift:9000"),
        "mcp_username":  _env("MCP_USERNAME", "admin"),
        "mcp_password":  _env("MCP_PASSWORD", "secret"),

        # ─── LLM Backend ───────────────────────────────────
        # openai    → uses OpenAI cloud
        # vllm      → self-hosted OAI-compatible
        # llama     → local llama.cpp / LM Studio
        "llm_backend":   _env("LLM_BACKEND", "openai").lower(),
        "llm_base_url":  _env("LLM_BASE_URL", ""),               # e.g. http://vllm:8000/v1
        "llm_model":     _env("LLM_MODEL", "gpt-4o-mini"),

        # ─── Data Layer ─────────────────────────────────────
        "redis_host": _env("REDIS_HOST", "redis"),
        "redis_port": int(_env("REDIS_PORT", "6379")),
        "redis_db": int(_env("REDIS_DB", "0")),
        "redis_ttl": int(_env("REDIS_TTL", str(60 * 60 * 24 * 7))),

        # ─── Embeddings ────────────────────────────────────
        # Used when backend is not OpenAI or overridden (e.g., for HuggingFace local/hub)
        "embedding_model": _env( # Kept this for non-Bedrock/OpenAI embedding models
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    }
