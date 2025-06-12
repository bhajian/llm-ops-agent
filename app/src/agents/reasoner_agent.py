"""
ReasonerAgent
-------------
Reads everything on the blackboard + the 250-token conversation summary and
produces the final answer.
"""

from typing import Iterator

from langchain.schema import AIMessage

from ..llm import get_llm
from ..integrations.tools import MemorySummaryTool
from ..integrations import blackboard


def run(
    user_query: str,
    conversation_id: str,
    *,
    stream: bool = False,
):
    llm = get_llm(streaming=stream)
    mem_tool = MemorySummaryTool()

    context = blackboard.read_all(conversation_id)
    summary = mem_tool.run(conversation_id)

    prompt = (
        "You are ReasonerAgent.\n"
        "Context provided by other agents (JSON):\n"
        f"{context}\n\n"
        f"Conversation summary:\n{summary}\n\n"
        "Using ONLY this context, answer the user's latest question clearly and "
        "concisely. Do not reveal internal data structures or previous dialogues.\n"
        f"User question: {user_query}"
    )

    if stream:
        for chunk in llm.stream(prompt):
            if isinstance(chunk, AIMessage):
                yield chunk.content
    else:
        resp = llm.invoke(prompt)
        return resp.content
