# app/llm.py
# ──────────────────────────────────────────────────────────────
# Unified factory for chat + embedding models
#
# Required env vars (examples):
#   • LLM_BACKEND         = bedrock | openai | vllm | local | gemini
#   • EMBEDDING_BACKEND   = bedrock | openai | hf | local | auto | gemini
#
# Bedrock extras (typical):\
#   • BEDROCK_REGION          us-east-2
#   • BEDROCK_PROFILE         bedrock-admin   (optional)
#   • BEDROCK_EMBED_MODEL     amazon.titan-embed-text-v2:0
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Set

from .config import get_settings  # ← relative import (project rule)

_cfg = get_settings()

# ──────────────── constants / sanity checks ────────────────
_VALID_LLM_BACKENDS: Set[str] = {"bedrock", "openai", "vllm", "local", "gemini"}
_VALID_EMB_BACKENDS: Set[str] = {"bedrock", "openai", "hf", "local", "auto", "gemini"}

def _env_choice(key: str, default: str, valid: Set[str]) -> str:
    """Lower-case env value and validate against allowed set."""
    val = (_cfg.get(key) or default).lower()
    if val not in valid:
        raise ValueError(f"{key.upper()} must be one of {sorted(valid)} — got '{val}'")
    return val


_BACKEND = _env_choice("llm_backend", "openai", _VALID_LLM_BACKENDS)
_EMBED_BACKEND = _env_choice("embedding_backend", "auto", _VALID_EMB_BACKENDS)

_MODEL = _cfg.get("llm_model") or "gpt-3.5-turbo"
_BASEURL = _cfg.get("llm_base_url") or None  # http://host:port/v1
_APIKEY = _cfg.get("openai_api_key")  # may be None for vLLM/local

# Bedrock
_BEDROCK_REGION = (
    _cfg.get("bedrock_region") or os.getenv("AWS_REGION") or "us-east-1"
)
_BEDROCK_PROFILE = _cfg.get("bedrock_profile") or os.getenv("AWS_PROFILE")
# Bedrock Embeddings Model Name from config
_BED_EMB_MODEL = _cfg.get("bedrock_embedding_model") or "amazon.titan-embed-text-v2:0"


# Embeddings (non-Bedrock specific)
_HF_EMB_ID = (_cfg.get("embedding_model") or "").strip() or (
    "sentence-transformers/all-MiniLM-L6-v2"
)
_HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
_LOCAL_EMB_PATH = _cfg.get("local_embedding_path")  # ./models/all-MiniLM (optional)

# ───────────────────────── Chat model factory ─────────────────────────
def get_llm(
    streaming: bool = False,
    callbacks: Optional[List] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 1024,
):
    """
    Return a LangChain Chat model for the active backend.

    Parameters
    ----------
    streaming   : bool   – enable token streaming
    callbacks   : list   – list[BaseCallbackHandler]
    temperature : float  – defaults to 0.7
    max_tokens  : int    – cap output length (avoids runaway costs)
    """
    temp = 0.7 if temperature is None else temperature
    common = dict(streaming=streaming, callbacks=callbacks or [], request_timeout=40)

    # ---------- OpenAI cloud -------------------------------------
    if _BACKEND == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=_MODEL,
            api_key=_APIKEY,
            base_url=_BASEURL,
            temperature=temp,
            max_tokens=max_tokens,
            **common,
        )

    # ---------- AWS Bedrock --------------------------------------
    if _BACKEND == "bedrock":
        try:
            from langchain_aws.chat_models import ChatBedrock
            from langchain_aws.llms import BedrockLLM 
            from botocore.config import Config
            from botocore.session import Session
        except ImportError as e:  
            sys.exit(
                f"❌  ChatBedrock/BedrockLLM missing or import path incorrect: {e}. "
                "Install with: pip install 'langchain-aws>=0.1.7' 'boto3>=1.34'"
            )

        session = None
        if _BEDROCK_PROFILE:
            session = Session(profile=_BEDROCK_PROFILE)

        client_config = Config(read_timeout=120)

        llm_model_kwargs = {"temperature": temp} 

        # FIX: Remove stop_sequences for Llama 3/4 if they are causing issues.
        # The error "Stop sequence key name for meta is not supported" means
        # even passing an empty list for this key is problematic for these specific models.
        # We will no longer conditionally add 'stop_sequences': []
        
        # Determine if it's a chat-specific model or a general text model
        if _MODEL.startswith("anthropic.claude") or _MODEL.startswith("meta.llama3") or _MODEL.startswith("meta.llama4"):
            # Use ChatBedrock for models that typically follow chat message structure
            return ChatBedrock(
                model_id=_MODEL,
                client=session.client("bedrock-runtime", region_name=_BEDROCK_REGION, config=client_config) if session else None,
                region_name=_BEDROCK_REGION,
                credentials_profile_name=_BEDROCK_PROFILE,
                streaming=streaming,
                callbacks=callbacks or [],
                model_kwargs=llm_model_kwargs # temperature is now inside llm_model_kwargs
            )
        else: # For other Bedrock text models not covered by above conditions
            return BedrockLLM( 
                model_id=_MODEL,
                client=session.client("bedrock-runtime", region_name=_BEDROCK_REGION, config=client_config) if session else None,
                region_name=_BEDROCK_REGION,
                credentials_profile_name=_BEDROCK_PROFILE,
                streaming=streaming,
                callbacks=callbacks or [],
                model_kwargs=llm_model_kwargs # temperature is now inside llm_model_kwargs
            )

    # ---------- Google Gemini (via LangChain) --------------------
    if _BACKEND == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            sys.exit(
                "❌  ChatGoogleGenerativeAI missing. Install with:\n"
                "    pip install 'langchain-google-genai>=0.0.1'"
            )
        return ChatGoogleGenerativeAI(
            model=_MODEL,
            temperature=temp,
            max_tokens=max_tokens,
            **common,
        )

    # ---------- vLLM / local OpenAI-compatible endpoints ---------
    if _BACKEND in {"vllm", "local"}:
        from langchain_community.chat_models import ChatOpenAI as CompatChat

        return CompatChat(
            model_name=_MODEL,
            base_url=_BASEURL,  # e.g. http://localhost:11434/v1
            api_key=_APIKEY or "NA",  # ignored by most self-hosted servers
            temperature=temp,
            max_tokens=max_tokens,
            **common,
        )

    raise RuntimeError(f"Unsupported LLM backend '{_BACKEND}'")


# ───────────────────────── Embedding factory ─────────────────────────
def get_embeddings():
    """
    Return an embeddings model according to EMBEDDING_BACKEND.

    Order when EMBEDDING_BACKEND=auto
    ---------------------------------
    • chat backend == openai   → OpenAIEmbeddings
    • chat backend == bedrock  → BedrockEmbeddings
    • otherwise                → HuggingFaceEmbeddings (hub / local)
    """

    def _auto_choice() -> str:
        if _BACKEND == "openai":
            return "openai"
        if _BACKEND == "bedrock":
            return "bedrock"
        if _BACKEND == "gemini":
            return "gemini"
        return "hf"

    backend = _EMBED_BACKEND if _EMBED_BACKEND != "auto" else _auto_choice()

    # ---------- OpenAI embeddings ---------------------------------
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    
    # ---------- Google Gemini embeddings --------------------------
    if backend == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            sys.exit(
                "❌  GoogleGenerativeAIEmbeddings missing. Install with:\n"
                "    pip install 'langchain-google-genai>=0.0.1'"
            )
        # Assuming the model can be configured via _MODEL, or a specific embedding model ID
        return GoogleGenerativeAIEmbeddings(
            model=_MODEL if "embedding" in _MODEL else "models/embedding-001",
        )

    # ---------- Bedrock embeddings --------------------------------
    if backend == "bedrock":
        try:
            from langchain_aws.embeddings import BedrockEmbeddings
        except ImportError:
            sys.exit(
                "❌  BedrockEmbeddings missing. Install with:\n"
                "    pip install 'langchain-aws>=0.1.7' 'boto3>=1.34'"
            )

        return BedrockEmbeddings(
            model_id=_BED_EMB_MODEL, # Use the specific bedrock embedding model ID from config
            region_name=_BEDROCK_REGION,
            credentials_profile_name=_BEDROCK_PROFILE,
        )

    # ---------- Local on-disk HF model ----------------------------
    if backend == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        path = Path(_LOCAL_EMB_PATH or _HF_EMB_ID).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Local embedding model not found: {path}.\n"
                "Set EMBEDDING_MODEL or LOCAL_EMBEDDING_PATH to a valid dir/file."
            )
        return HuggingFaceEmbeddings(model_name=str(path), cache_folder=str(path))

    # ---------- HuggingFace hub embeddings ------------------------
    if backend == "hf":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if _HF_TOKEN:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _HF_TOKEN)

        try:
            return HuggingFaceEmbeddings(model_name=_HF_EMB_ID)
        except OSError as err:
            print(
                f"⚠️  HF load failed for '{_HF_EMB_ID}', "
                f"falling back to MiniLM. ({err})"
            )
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    raise RuntimeError(f"Unsupported EMBEDDING_BACKEND '{backend}'")
