from typing import Any, ClassVar, Optional

from langchain.tools import BaseTool

from ..weaviate_client import WeaviateVectorClient


class VectorSearchTool(BaseTool):
    name: ClassVar[str] = "vector_search"
    description: ClassVar[str] = (
        "Hybrid semantic/BM25 search over RAG docs. "
        "Args: query(str), top_k(int). Returns list of chunks."
    )

    def __init__(self, client: Optional[WeaviateVectorClient] = None):
        super().__init__()
        self.client = client or WeaviateVectorClient()

    def _run(self, query: str, top_k: int = 5, **_: Any) -> str:
        res = self.client.search(query, top_k=top_k)
        return str([{"text": r["text"], "score": r["score"]} for r in res])

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported for VectorSearchTool.")
