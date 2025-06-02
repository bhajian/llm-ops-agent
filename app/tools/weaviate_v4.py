# app/tools/weaviate_v4.py

from urllib.parse import urlparse
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Property, DataType, Configure


def connect_weaviate(url: str) -> WeaviateClient:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    secure = parsed.scheme == "https"

    return WeaviateClient(
        connection_params=ConnectionParams(
            http={"host": host, "port": port, "secure": secure},
            grpc={"host": host, "port": 50051, "secure": secure}
        )
    )


def ensure_document_class(client: WeaviateClient, class_name: str = "Document"):
    try:
        client.collections.get(class_name)
        print(f"[SCHEMA] Collection '{class_name}' already exists.")
    except Exception:
        print(f"[SCHEMA] Creating collection '{class_name}'...")
        client.collections.create(
            name=class_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
