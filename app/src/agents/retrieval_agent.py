"""
RetrievalAgent
--------------
Finds relevant passages from the RAG vector store and returns a concise summary.
"""

from typing import Iterator

from langchain.schema import AIMessage

from ..llm import get_llm
from ..integrations.tools import VectorSearchTool
from ..integrations import blackboard


def run(
    user_query: str,
    conversation_id: str,
    *,
    stream: bool = False,
):
    llm = get_llm(streaming=stream)
    tool = VectorSearchTool()

    prompt = (
        "You are RetrievalAgent. Use the `vector_search` tool to fetch "
        "relevant chunks, then summarise them in 4–6 bullet points.\n"
        f"User question: {user_query}"
    )

    if stream:
        for chunk in llm.stream(prompt, tools=[tool]):
            if isinstance(chunk, AIMessage):
                yield chunk.content
    else:
        resp = llm.invoke(prompt, tools=[tool])
        blackboard.write(conversation_id, "retrieval", {"answer": resp.content})
        return resp.content
