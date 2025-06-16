# app/tools/weaviate_v4.py
"""
Unifies Weaviate client access across SDK v3 and v4.

Exports
    connect_weaviate()      → returns a connected client
    ensure_document_class() → creates a simple "Document" class if missing

The rest of the codebase can keep importing:

    from app.tools.weaviate_v4 import connect_weaviate, ensure_document_class
"""
from __future__ import annotations
import os, logging, inspect
from typing import Any

_WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
_WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

# --------------------------------------------------------------------------- #
#  Detect SDK version
# --------------------------------------------------------------------------- #
try:
    # v4: main entry is `WeaviateClient`
    from weaviate import WeaviateClient as _W4Client
    import weaviate
    _IS_V4 = True
except ImportError:
    # v3 fallback
    import weaviate                                   # type: ignore
    _IS_V4 = False


# --------------------------------------------------------------------------- #
#  Connection helpers
# --------------------------------------------------------------------------- #
def _make_v4_client() -> "weaviate.WeaviateClient":          # pragma: no cover
    auth = (
        weaviate.AuthApiKey(_WEAVIATE_API_KEY) if _WEAVIATE_API_KEY else None
    )
    return _W4Client(
        url=_WEAVIATE_URL,
        auth_credentials=auth,
        timeout=(5, 60),
        skip_init_check=True,
    )


def _make_v3_client() -> "weaviate.Client":                 # pragma: no cover
    auth = (
        weaviate.AuthApiKey(_WEAVIATE_API_KEY) if _WEAVIATE_API_KEY else None
    )
    return weaviate.Client(
        url=_WEAVIATE_URL,
        auth_client_secret=auth,
        timeout_config=(5, 60),
    )


def connect_weaviate():
    """
    Return a connected Weaviate client (v3 or v4 depending on what's installed).
    """
    if _IS_V4:
        logging.info("weaviate-v4 client detected")
        return _make_v4_client()
    logging.info("weaviate-v3 client detected")
    return _make_v3_client()


# --------------------------------------------------------------------------- #
#  Schema bootstrap
# --------------------------------------------------------------------------- #
def ensure_document_class(
    client: Any,
    class_name: str = "Document",
    text_key: str = "text",
):
    """
    Ensure a simple text-only class exists. Idempotent for both SDK versions.
    """
    if _IS_V4:
        # v4 collections API
        if class_name in [c.name for c in client.collections.list_all()]:
            return
        client.collections.create(
            name=class_name,
            properties=[{"name": text_key, "dataType": ["text"]}],
            vectorizer_config={"vectorizer": "none"},
        )
    else:
        # v3 schema API
        schema = client.schema.get()
        if any(c.get("class") == class_name for c in schema.get("classes", [])):
            return
        client.schema.create_class(
            {
                "class": class_name,
                "description": "Generic document chunk",
                "vectorizer": "none",
                "properties": [
                    {
                        "name": text_key,
                        "dataType": ["text"],
                        "description": "chunk text",
                    }
                ],
            }
        )
