# app/llm.py
"""
LLM / Embedding factory helpers
────────────────────────────────
Embeddings
  • Hugging Face sentence-transformers  (EMBEDDING_BACKEND=hf)
  • OpenAI /text-embedding-3            (EMBEDDING_BACKEND=openai)
  • Local mean-pool stub                (EMBEDDING_BACKEND=local)

LLMs
  • Any OpenAI-compatible chat endpoint (OpenAI, vLLM, llama.cpp, Bedrock)

Agents import `chat_llm(messages)`.
"""
from __future__ import annotations
import os, inspect, asyncio
from functools import lru_cache
from typing import List, Dict, Literal, Optional

from dotenv import load_dotenv
load_dotenv(override=False)

# ───────────────────────────  Embeddings  ───────────────────────────
_EmbedBackend = Literal["hf", "openai", "local"]

# Hugging Face ------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login as _hf_login

if tok := os.getenv("HUGGINGFACE_HUB_TOKEN"):
    # avoid `git` dependency inside slim image
    _hf_login(tok, add_to_git_credential=False)

def _hf_embed(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

# OpenAI ------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings

def _openai_embed(model_name: str):
    return OpenAIEmbeddings(
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
    )

# Local stub (mean-pool) -------------------------------------------
class _MeanPoolEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(hash(t) % 1000)] for t in texts]
    def embed_query(self, text: str) -> List[float]:
        return [float(hash(text) % 1000)]

def _local_embed(_: str):
    return _MeanPoolEmbeddings()

def get_embeddings(model_name: Optional[str] = None):
    backend: _EmbedBackend = os.getenv("EMBEDDING_BACKEND", "hf").lower()
    if backend == "hf":
        model = model_name or os.getenv("HF_EMBED_MODEL",
                                        "sentence-transformers/all-MiniLM-L6-v2")
        return _hf_embed(model)
    if backend == "openai":
        model = model_name or os.getenv("OPENAI_EMBED_MODEL",
                                        "text-embedding-3-small")
        return _openai_embed(model)
    return _local_embed("")


# ─────────────────────────────  LLMs  ──────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

_BackendT = Literal["openai"]

def _openai_llm(temp: float, max_tokens: int, streaming: bool):
    return ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=temp,
        max_tokens=max_tokens,
        streaming=streaming,
    )

_BACKENDS: dict[_BackendT, callable] = {"openai": _openai_llm}

@lru_cache(maxsize=None)
def get_llm(temperature=0.2, max_tokens=1024, streaming=True):
    backend: _BackendT = os.getenv("LLM_BACKEND", "openai").lower()
    return _BACKENDS[backend](temperature, max_tokens, streaming)


# ─────────────────────  async chat wrapper  ────────────────────────
def _dict2msg(d: Dict[str, str]):
    role, content = d["role"], d.get("content") or d.get("msg", "")
    if role == "user":      return HumanMessage(content)
    if role == "assistant": return AIMessage(content)
    return SystemMessage(content)

async def _default_chat_llm(messages: List[Dict[str, str]]) -> str:
    llm_msgs = [_dict2msg(m) for m in messages]
    llm = get_llm(streaming=False)
    res = await llm.ainvoke(llm_msgs)
    return res.content.strip()

_chat_fn = globals().get("chat_llm", _default_chat_llm)

async def chat_llm(messages: List[Dict[str, str]]) -> str:
    out = _chat_fn(messages)
    return await out if inspect.isawaitable(out) else out
