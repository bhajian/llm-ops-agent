# app/config.py
"""
Single source of truth for runtime settings.

▪ Works out-of-the-box with OpenAI cloud
▪ Switch to vLLM / llama by exporting
     LLM_BACKEND=vllm
     LLM_BASE_URL=http://vllm:8000/v1
     LLM_MODEL=llama-3-8b-instruct
"""

import os
from functools import lru_cache


def _env(key: str, default: str | None = None) -> str | None:
    """Tiny helper for env reads (keeps typing happy)."""
    return os.getenv(key, default)


@lru_cache
def get_settings() -> dict:
    return {
        # ─── data layer ───────────────────────────────────────
        "redis_host":    _env("REDIS_HOST", "redis"),
        "weaviate_url":  _env("WEAVIATE_URL", "http://weaviate:8080"),

        # ─── façade / MCPs ───────────────────────────────────
        "mcp_fmp_url":   _env("MCP_FMP_URL",  "http://mcp-fmp:9000"),
        "mcp_k8s_url":   _env("MCP_K8S_URL",  "http://mcp-openshift:9000"),
        "mcp_username":  _env("MCP_USERNAME", "admin"),
        "mcp_password":  _env("MCP_PASSWORD", "secret"),

        # ─── LLM & embeddings ────────────────────────────────
        #   openai    → cloud ChatCompletion / embeddings
        #   vllm      → any OAI-compatible local server
        #   llama     → llama.cpp, LM Studio, etc.
        "llm_backend":  _env("LLM_BACKEND", "openai").lower(),
        "llm_base_url": _env("LLM_BASE_URL", ""),          # e.g. http://vllm:8000/v1
        "llm_model":    _env("LLM_MODEL", "gpt-4o-mini"),
        # HF model used when backend != openai, or as override
        "embedding_model": _env(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    }
