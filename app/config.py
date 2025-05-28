# --- app/config.py ---
import os
from dotenv import load_dotenv
load_dotenv()

def get_settings():
    return {
        "llm_backend": os.getenv("LLM_BACKEND", "lmstudio"),
        "llm_base_url": os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        "weaviate_url": os.getenv("WEAVIATE_URL", "http://localhost:8080")
    }
