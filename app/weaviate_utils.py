# app/weaviate_utils.py
"""
Auto-create the `Document` class using **weaviate-client v3** API.
Safe to call multiple times.
"""

import weaviate


def ensure_document_class(client: weaviate.Client) -> None:
    if client.schema.contains({"class": "Document"}):
        return

    doc_schema = {
        "class": "Document",
        "description": "Chunks for Retrieval-QA",
        "vectorizer": "text2vec-openai",           # or text2vec-huggingface
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Raw text chunk",
            }
        ],
    }
    client.schema.create_class(doc_schema)
