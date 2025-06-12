"""
Ensure Weaviate 'Document' class exists.
Runs once at FastAPI startup.
"""

import weaviate
from ..config import get_settings

_cfg = get_settings()
client = weaviate.Client(_cfg.weaviate_url)    # ← positional arg

def ensure_schema() -> None:
    schema = client.schema.get()
    if any(c["class"] == "Document" for c in schema.get("classes", [])):
        return

    doc_class = {
        "class": "Document",
        "vectorizer": "none",
        "properties": [{"name": "text", "dataType": ["text"]}],
    }
    client.schema.create_class(doc_class)
    print("✅  Weaviate 'Document' class created")
