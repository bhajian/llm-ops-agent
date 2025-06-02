# app/weaviate_utils.py  – replace the file
import weaviate

_DOC_SCHEMA = {
    "class": "Document",
    "description": "Chunks for Retrieval-QA",
    "vectorizer": "text2vec-openai",          # or text2vec-huggingface
    "properties": [
        {
            "name": "content",
            "description": "Raw text chunk",
            "dataType": ["text"],
        }
    ],
}

def ensure_document_class(client: weaviate.Client) -> None:
    """
    Create the `Document` class if it doesn't exist (v3-style API, safe-idempotent).
    """
    for c in client.schema.get().get("classes", []):
        if c["class"] == "Document":
            return                       # already present

    client.schema.create_class(_DOC_SCHEMA)
