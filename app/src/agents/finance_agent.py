"""
FinanceAgent
------------
Answers questions requiring live or historical market data via MCP.
"""

from typing import Iterator

from langchain.schema import AIMessage

from ..llm import get_llm
from ..integrations.tools import MarketDataTool
from ..integrations import blackboard


def run(
    user_query: str,
    conversation_id: str,
    *,
    stream: bool = False,
):
    """Return full answer (string) or yield chunks when stream=True."""
    llm = get_llm(streaming=stream)
    tool = MarketDataTool()

    prompt = (
        "You are FinanceAgent, an analyst with direct access to market data.\n"
        "When helpful, call the `market_data` tool; otherwise answer directly.\n"
        f"User question: {user_query}"
    )

    if stream:
        for chunk in llm.stream(prompt, tools=[tool]):
            if isinstance(chunk, AIMessage):
                yield chunk.content
    else:
        resp = llm.invoke(prompt, tools=[tool])
        # persist for ReasonerAgent
        blackboard.write(conversation_id, "finance", {"answer": resp.content})
        return resp.content
