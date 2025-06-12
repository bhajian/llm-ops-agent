"""
LLM + Embedding factory.

Supports:
• Chat back-ends: openai | bedrock | vllm | local
• Embedding back-ends: bedrock | openai | hf | local | auto

Switch providers purely via environment variables / .env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Set

from .config import get_settings

_cfg = get_settings()

# ───────── Allowed values & helpers ─────────
_VALID_LLM: Set[str] = {"bedrock", "openai", "vllm", "local"}
_VALID_EMB: Set[str] = {"bedrock", "openai", "hf", "local", "auto"}


def _choice(key: str, default: str, pool: Set[str]) -> str:
    val = (_cfg.dict().get(key) or default).lower()
    if val not in pool:
        raise ValueError(f"{key.upper()} must be one of {sorted(pool)} (got '{val}')")
    return val


_BACKEND = _choice("llm_backend", "openai", _VALID_LLM)
_EMBED_BACKEND = _choice("embedding_backend", "auto", _VALID_EMB)

_MODEL = _cfg.llm_model
_BASEURL = _cfg.llm_base_url
_APIKEY = _cfg.openai_api_key

# Bedrock settings
_BED_REGION = _cfg.bedrock_region
_BED_PROFILE = _cfg.bedrock_profile # This is still here for reference but not used in the BedrockEmbeddings constructor directly

# Embedding settings
_HF_MODEL = _cfg.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
_HF_TOKEN = _cfg.huggingface_api_token
_LOCAL_EMB_PATH = _cfg.local_embedding_path


# ────────────────── LLM factory ──────────────────
def get_llm(
    *, streaming: bool = False, temperature: float = 0.7, max_tokens: int = 1024
):
    """
    Return LangChain LLM.  Only instantiate once (per set of args).
    """
    if _BACKEND == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=_MODEL,
            base_url=_BASEURL,
            api_key=_APIKEY,
            temperature=temperature,
            streaming=streaming,
            max_tokens=max_tokens,
        )

    if _BACKEND == "bedrock":
        try:
            from langchain_aws.chat_models import ChatBedrock
        except ImportError:
            sys.exit(
                "❌  `langchain-aws` missing. Install with:\\n"
                "    pip install 'langchain-aws>=0.1.7' 'boto3>=1.34'"
            )

        return ChatBedrock(
            model_id=_MODEL,
            region_name=_BED_REGION,
            credentials_profile_name=_BED_PROFILE, # This can remain if ChatBedrock supports it, but the error was with Embeddings
            streaming=streaming,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if _BACKEND == "vllm":
        try:
            from langchain_community.llms import VLLM
        except ImportError:
            sys.exit(
                "❌  `vllm` missing. Install with:\\n"
                "    pip install 'vllm>=0.2.7' 'torch>=2.1'"
            )

        return VLLM(
            model=_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            vllm_kwargs={"disable_sliding_window": True},  # for Mixtral
        )

    if _BACKEND == "local":
        try:
            from langchain_community.llms import LlamaCpp
        except ImportError:
            sys.exit(
                "❌  `llama-cpp-python` missing. Install with:\\n"
                "    pip install 'llama-cpp-python>=0.2.37' 'huggingface_hub>=0.20'"
            )

        return LlamaCpp(
            model_path=str(Path(_MODEL).expanduser()),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM backend: {_BACKEND}")


# ────────────────── Embedding factory ──────────────────
def get_embeddings():
    """
    Return LangChain Embeddings.
    """
    backend = _EMBED_BACKEND
    if backend == "auto":
        if _BACKEND == "openai":
            backend = "openai"
        elif _BACKEND == "bedrock":
            backend = "bedrock"
        else:  # vllm | local
            backend = "hf"

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()

    if backend == "bedrock":
        try:
            from langchain_aws.embeddings import BedrockEmbeddings
        except ImportError:
            sys.exit(
                "❌  `langchain-aws` missing. Install with:\\n"
                "    pip install 'langchain-aws>=0.1.7' 'boto3>=1.34'"
            )

        return BedrockEmbeddings(
            model_id=_cfg.bedrock_embedding_model,
            region_name=_BED_REGION,
            # REMOVED: credentials_profile_name=_BED_PROFILE, # <-- REMOVE THIS LINE
        )

    if backend == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        path = Path(_LOCAL_EMB_PATH or _HF_MODEL).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Local embedding model not found: {path}\\n"
                "Set EMBEDDING_MODEL or LOCAL_EMBEDDING_PATH."
            )
        return HuggingFaceEmbeddings(model_name=str(path), cache_folder=str(path))

    if backend == "hf":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if _HF_TOKEN:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _HF_TOKEN)

        try:
            return HuggingFaceEmbeddings(model_name=_HF_MODEL)
        except OSError:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    raise RuntimeError(f"Unknown embedding backend: {backend}")
