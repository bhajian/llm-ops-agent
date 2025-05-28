# --- app/llm.py ---
import os                                               # ➜ was missing
from app.config import get_settings
from langchain_openai import ChatOpenAI

def get_llm(*, streaming: bool = False, callbacks=None):
    """
    Return a ChatOpenAI runnable. Pass streaming=True to enable token streaming.
    """
    cfg = get_settings()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return ChatOpenAI(
        model_name=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=streaming,
        callbacks=callbacks or [],
    )
