# app/llm.py
"""
Central helper to get an OpenAI-compatible LLM runnable.
Works in both streaming and non-stream modes.
"""

import os
from langchain_openai import ChatOpenAI


def get_llm(*, streaming: bool = False, callbacks=None):
    """
    Parameters
    ----------
    streaming : bool
        If True, the model will send tokens incrementally.
    callbacks : list[BaseCallbackHandler] | None
        Optional LangChain callback handlers.

    Returns
    -------
    ChatOpenAI
        Configured model object.
    """
    return ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=streaming,
        callbacks=callbacks or [],
    )
