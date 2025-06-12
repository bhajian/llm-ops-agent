"""
K8sAgent
--------
Handles simple Kubernetes/OpenShift operations, such as scaling a deployment
or listing pods.
"""

from typing import Iterator

from langchain.schema import AIMessage

from ..llm import get_llm
from ..integrations.tools import K8sTool


def run(
    user_query: str,
    conversation_id: str,  # unused but keeps signature consistent
    *,
    stream: bool = False,
):
    llm = get_llm(streaming=stream)
    k8s_tool = K8sTool()

    prompt = (
        "You are K8sAgent, responsible for basic Kubernetes operations.\n"
        "If the user asks to scale or inspect resources, call the `k8s` tool.\n"
        f"User question: {user_query}"
    )

    if stream:
        for chunk in llm.stream(prompt, tools=[k8s_tool]):
            if isinstance(chunk, AIMessage):
                yield chunk.content
    else:
        resp = llm.invoke(prompt, tools=[k8s_tool])
        return resp.content
