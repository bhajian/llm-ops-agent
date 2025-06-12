"""
src/config.py
─────────────
Centralised settings loader for the whole project.

• Reads .env automatically (if present)
• Uses pydantic-settings (v2) for environment validation
• Expose a singleton via get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings  # ← new import for Pydantic v2

# Auto-load .env if it exists next to docker-compose.yaml
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)  # noqa: S603  (explicit path OK)

# ─────────────────────────── Settings model ────────────────────────────
class Settings(BaseSettings):
    # LLM back-end
    llm_backend: str = Field("openai", description="openai | bedrock | vllm | local")
    llm_model: str = Field("gpt-3.5-turbo")
    llm_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Bedrock
    bedrock_region: str = Field("us-east-1")
    bedrock_profile: Optional[str] = None
    bedrock_embedding_model: str = Field("amazon.titan-embed-text-v2:0")

    # Embeddings
    embedding_backend: str = Field("auto")  # bedrock | openai | hf | local | auto
    embedding_model: str = Field("")
    local_embedding_path: Optional[str] = None
    huggingface_api_token: Optional[str] = None # <-- ADD THIS LINE

    # External services
    mcp_base_url: Optional[str] = None
    mcp_api_key: Optional[str] = None
    weaviate_url: str = "http://weaviate:8080"
    redis_url: str = "redis://redis:6379/0"

    # LangSmith (optional)
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None

    @validator("llm_backend", pre=True)
    def _validate_llm_backend(cls, v: str) -> str:
        v = v.lower()
        if v not in {"openai", "bedrock", "vllm", "local"}:
            raise ValueError(f"LLM_BACKEND must be one of openai | bedrock | vllm | local (got {v})")
        return v

    @validator("embedding_backend", pre=True)
    def _validate_embedding_backend(cls, v: str) -> str:
        v = v.lower()
        if v not in {"bedrock", "openai", "hf", "local", "auto"}:
            raise ValueError(f"EMBEDDING_BACKEND must be one of bedrock | openai | hf | local | auto (got {v})")
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Return cached Settings object.
    """
    return Settings()
