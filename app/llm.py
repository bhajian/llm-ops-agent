# app/llm.py

from typing import Literal, Optional
from app.config import get_settings

_cfg = get_settings()
_BACKEND = _cfg["llm_backend"]  # openai | vllm | llama

# -------- chat model --------
def get_llm(streaming: bool = False, callbacks=None, temperature: Optional[float] = None):
    """
    Returns a Chat model instance based on backend config.
    Supports OpenAI, vLLM, LM Studio, etc.
    """
    model = _cfg["llm_model"]
    base_url = _cfg.get("llm_base_url")
    temp = temperature if temperature is not None else 0.7

    if _BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temp,
            streaming=streaming,
            callbacks=callbacks,
            api_key=_cfg.get("openai_api_key"),
            base_url=base_url or None
        )

    # vLLM / LM Studio / llama.cpp (OpenAI-compatible)
    from langchain_community.chat_models import ChatOpenAI as CompatChat
    return CompatChat(
        model_name=model,
        base_url=base_url,
        api_key="NA",
        temperature=temp,
        streaming=streaming,
        callbacks=callbacks
    )

# -------- embedding model --------
def get_embeddings():
    """
    Returns an embedding model instance based on config.
    Defaults to HuggingFace if OpenAI not explicitly configured.
    """
    try:
        if _BACKEND == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
    except Exception:
        pass

    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=_cfg["embedding_model"])
