from typing import List

import weaviate
from ..config import get_settings
from ..llm import get_embeddings

_cfg = get_settings()

class WeaviateVectorClient:
    def __init__(self):
        self.client = weaviate.Client(_cfg.weaviate_url)      # ← positional
        self.embeddings = get_embeddings()

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        vec = self.embeddings.embed_query(query)
        res = (
            self.client.query.get("Document", ["text"])
            .with_near_vector({"vector": vec})
            .with_limit(top_k)
            .do()
        )
        return [
            {"text": item["text"], "score": item["_additional"]["certainty"]}
            for item in res["data"]["Get"]["Document"]
        ]
