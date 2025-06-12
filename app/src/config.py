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

    # External services
    mcp_base_url: Optional[str] = None
    mcp_api_key: Optional[str] = None
    weaviate_url: str = "http://weaviate:8080"
    redis_url: str = "redis://redis:6379/0"

    # LangSmith (optional)
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None

    @validator("llm_backend", "embedding_backend")
    def _lower(cls, v: str) -> str:  # noqa: N805
        return v.lower()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance (singleton)."""
    return Settings()
