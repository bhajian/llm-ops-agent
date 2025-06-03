# app/llm.py
from __future__ import annotations

import os
from typing import Optional, List

from app.config import get_settings

_cfg = get_settings()

# ───────────────────────── env / config ─────────────────────────
_BACKEND   = (_cfg.get("llm_backend")      or "openai").lower()        # openai | vllm | llama
_MODEL     =  _cfg.get("llm_model")        or "gpt-3.5-turbo"
_BASEURL   =  _cfg.get("llm_base_url")     or None                     # http://host:port/v1
_APIKEY    =  _cfg.get("openai_api_key")                               # may be None for vLLM

# Embeddings
_EMBED_BACKEND = (_cfg.get("embedding_backend") or "auto").lower()     # auto | openai | hf
_HF_EMB_ID     =  _cfg.get("embedding_model", "").strip() \
                 or "sentence-transformers/all-MiniLM-L6-v2"
_HF_TOKEN      =  os.getenv("HUGGINGFACE_HUB_TOKEN")                   # new

# ───────────────────────── chat factory ──────────────────────────
def get_llm(
    streaming: bool = False,
    callbacks: Optional[List] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 1024,
):
    """
    Return a LangChain Chat model.

    Parameters
    ----------
    streaming   : enable token streaming callbacks
    callbacks   : list[BaseCallbackHandler]
    temperature : defaults to 0.7
    max_tokens  : cap output length (avoids 200k-TPM explosions)
    """
    temp = 0.7 if temperature is None else temperature
    common = dict(
        temperature=temp,
        streaming=streaming,
        callbacks=callbacks or [],
        max_tokens=max_tokens,
        request_timeout=40,
    )

    if _BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=_MODEL, api_key=_APIKEY, base_url=_BASEURL, **common)

    # Any OpenAI-compatible endpoint (vLLM, LM-Studio, llama.cpp, etc.)
    from langchain_community.chat_models import ChatOpenAI as CompatChat
    return CompatChat(
        model_name=_MODEL,
        base_url=_BASEURL,              # e.g. http://18.117.160.182:8000/v1
        api_key=_APIKEY or "NA",        # vLLM ignores the string but LangChain expects one
        **common,
    )

# ─────────────────────── embedding factory ───────────────────────
def get_embeddings():
    """
    Returns an embeddings model.

    Priority
    --------
    1. EMBEDDING_BACKEND=openai            → OpenAIEmbeddings
    2. (auto & chat backend == openai)     → OpenAIEmbeddings
    3. Otherwise                           → HuggingFaceEmbeddings
                                            (token picked up from env)
    """
    use_openai = (
        _EMBED_BACKEND == "openai"
        or (_EMBED_BACKEND == "auto" and _BACKEND == "openai")
    )

    if use_openai and _APIKEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    # ---------- Hugging Face path ----------
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # honour env-var token; no kw-arg needed
    if _HF_TOKEN:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _HF_TOKEN)

    model_id = (
        _HF_EMB_ID
        if _HF_EMB_ID.lower() != "openai"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        return HuggingFaceEmbeddings(model_name=model_id)
    except OSError as err:
        print(
            f"⚠️  Could not load HF embeddings '{model_id}'. "
            f"Falling back to MiniLM. ({err})"
        )
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
