"""
Generic LLM / Embedding factory
────────────────────────────────
Chat LLM
────────
  LLM_BACKEND = openai | vllm | bedrock            (default: openai)
  MODEL_NAME  = meta-llama/Meta-Llama-3-8B-Instruct

  OPENAI_API_BASE … OpenAI cloud   (chat)
  VLLM_BASE        … vLLM endpoint (chat or embeds)
  BEDROCK_REGION   … us‑east‑1 …
  OPENAI_API_KEY   … required syntactically for every OpenAI‑style client
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY … for Bedrock

Embeddings
──────────
  EMBEDDING_BACKEND = openai | vllm | hf           (default: openai)

  OPENAI_EMBED_MODEL = text-embedding-3-small
  OPENAI_EMBED_BASE  = <url>  (falls back to OPENAI_API_BASE)

  HF_EMBED_MODEL = sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import os
import inspect
from functools import lru_cache
from typing import Dict, Literal, List, Optional

from dotenv import load_dotenv

load_dotenv(override=False)

##############################################################################
# Helpers
##############################################################################

def _norm_base(url: str) -> str:
    """strip trailing / and a possible *duplicate* /v1 suffix"""
    url = url.rstrip("/")
    if url.lower().endswith("/v1"):
        url = url[:-3]  # drop the /v1
    return url

##############################################################################
# ────────────────────────────  EMBEDDINGS  ──────────────────────────────── #
##############################################################################
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def _openai_embeddings() -> OpenAIEmbeddings:
    base_url = _norm_base(
        os.getenv("OPENAI_EMBED_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com"))
    )
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        base_url=f"{base_url}/v1",
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )


def _vllm_embeddings() -> OpenAIEmbeddings:
    base_url = _norm_base(os.getenv("VLLM_BASE", "http://localhost:8000"))
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        base_url=f"{base_url}/v1",
        api_key="dummy",
    )


def _hf_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )


_EMBED_BACKENDS: Dict[str, callable] = {
    "openai": _openai_embeddings,
    "vllm": _vllm_embeddings,
    "hf": _hf_embeddings,
}


def get_embeddings():
    backend = os.getenv("EMBEDDING_BACKEND", "openai").lower()
    try:
        return _EMBED_BACKENDS[backend]()
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported EMBEDDING_BACKEND={backend!r}") from exc


##############################################################################
# ───────────────────────────────  CHAT  LLM  ────────────────────────────── #
##############################################################################
from langchain_openai import ChatOpenAI

_BackendT = Literal["openai", "vllm", "bedrock"]


def _openai_llm(temp: float, max_tokens: int, streaming: bool):
    base = _norm_base(os.getenv("OPENAI_API_BASE", "https://api.openai.com"))
    return ChatOpenAI(
        base_url=f"{base}/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        temperature=temp,
        max_tokens=max_tokens,
        streaming=streaming,
    )


def _vllm_llm(temp: float, max_tokens: int, streaming: bool):
    base = _norm_base(os.getenv("VLLM_BASE", "http://localhost:8000"))
    return ChatOpenAI(
        base_url=f"{base}/v1",
        api_key="dummy",  # vLLM ignores auth
        model=os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
        temperature=temp,
        max_tokens=max_tokens,
        streaming=streaming,
    )


def _bedrock_llm(temp: float, max_tokens: int, streaming: bool):
    try:
        from langchain_aws import ChatBedrock
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install langchain-aws to use Bedrock backend") from e

    return ChatBedrock(
        model_id=os.getenv("MODEL_NAME", "anthropic.claude-3-sonnet-20240229-v1:0"),
        region_name=os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1")),
        streaming=streaming,
        model_kwargs={"temperature": temp, "max_tokens": max_tokens},
    )


_LLM_BACKENDS: Dict[_BackendT, callable] = {
    "openai": _openai_llm,
    "vllm": _vllm_llm,
    "bedrock": _bedrock_llm,
}


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.2, max_tokens: int = 1024, streaming: bool = True):
    backend: _BackendT = os.getenv("LLM_BACKEND", "openai").lower()  # default
    try:
        return _LLM_BACKENDS[backend](temperature, max_tokens, streaming)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported LLM_BACKEND={backend!r}") from exc


##############################################################################
# ──────────────────  LIGHT‑WEIGHT async chat convenience  ───────────────── #
##############################################################################
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def _dict2lc(m: Dict[str, str]):
    role = m["role"]
    content = m.get("content", m.get("msg", ""))
    if role == "user":
        return HumanMessage(content)
    if role == "assistant":
        return AIMessage(content)
    return SystemMessage(content)  # treat anything else as system


async def chat_llm(messages: List[Dict[str, str]]) -> str:
    """Async helper used by agents – converts plain dicts → LC messages."""
    lc_msgs = [_dict2lc(m) for m in messages]
    llm = get_llm(streaming=False)
    result = await llm.ainvoke(lc_msgs)
    return result.content.strip()
