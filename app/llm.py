# app/llm.py
from typing import Literal
from app.config import get_settings

_cfg = get_settings()
_BACKEND = _cfg["llm_backend"]          # openai | vllm | llama


# -------- chat model --------
def get_llm(streaming: bool = False, callbacks=None):
    if _BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=_cfg["llm_model"],
            streaming=streaming,
            callbacks=callbacks,
            temperature=0,
        )

    # vLLM / llama.cpp / LM Studio (OpenAI-compatible)
    from langchain_community.chat_models import ChatOpenAI as CompatChat
    return CompatChat(
        model_name=_cfg["llm_model"],
        base_url=_cfg["llm_base_url"],
        api_key="NA",
        streaming=streaming,
        callbacks=callbacks,
        temperature=0,
    )


# -------- embedding model --------
def get_embeddings():
    if _BACKEND == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
        except Exception:
            pass  # fall through to HF

    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=_cfg["embedding_model"])
