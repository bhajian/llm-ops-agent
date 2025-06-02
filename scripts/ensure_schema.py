# scripts/ensure_schema.py

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Property, DataType, Configure


def ensure_document_class():
    client = WeaviateClient(
        connection_params=ConnectionParams(
            http={"host": "localhost", "port": 8080, "secure": False},
            grpc={"host": "localhost", "port": 50051, "secure": False}
        ),
        skip_init_checks=True  # ✅ skip gRPC health checks
    )

    client.connect()
    class_name = "Document"
    try:
        client.collections.get(class_name)
        print(f"✅ Class '{class_name}' already exists.")
    except Exception:
        print(f"🆕 Creating class '{class_name}'...")
        client.collections.create(
            name=class_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT)
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
        print("✅ Schema created.")

    client.close()


if __name__ == "__main__":
    ensure_document_class()
