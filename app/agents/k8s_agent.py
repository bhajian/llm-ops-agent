# app/agents/k8s_agent.py
# ────────────────────────────────────────────────────────────
"""
Functions for OpenShift Kubernetes cluster management, designed to be
used as nodes or helpers within the LangGraph orchestrator.
Includes tool invocation and answer synthesis using LLMs.
"""
from __future__ import annotations

import asyncio
import json
import re
import warnings
from typing import Any, Dict, Optional, List

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

# Local imports from common directory and tools
from app.llm import get_llm
from app.tools.openshift_tools import scale_deployment, get_pods # Import direct tool functions


# ─── Helper for extracting K8s arguments from user input ───
class ScaleDeploymentArgs(BaseModel):
    deployment_name: str = Field(..., description="The name of the OpenShift deployment.")
    replicas: int = Field(..., description="The number of replicas to scale to.")
    namespace: str = Field("default", description="The namespace of the deployment (default: 'default').")

class GetPodsArgs(BaseModel):
    namespace: str = Field("default", description="The namespace to list pods from (default: 'default').")

# We'll use a general LLM to decide which K8s tool to call and extract args
class K8sToolCall(BaseModel):
    tool_name: str = Field(..., description="The name of the K8s tool to call (e.g., 'scale_deployment', 'get_pods').")
    tool_args: Dict[str, Any] = Field({}, description="A dictionary of arguments for the tool call.")

_k8s_tool_parser = PydanticOutputParser(pydantic_object=K8sToolCall)
_k8s_extraction_llm = get_llm(temperature=0, streaming=False) # Non-streaming for structured output

_K8S_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert at identifying OpenShift Kubernetes management tasks and extracting their parameters. "
        "Your task is to determine which tool to call and what arguments to use based on the user's request. "
        "Available tools: "
        "- `scale_deployment(deployment_name: str, replicas: int, namespace: str = 'default')`: Scales an OpenShift deployment to a specified number of replicas."
        "- `get_pods(namespace: str = 'default')`: Lists pods in a specified namespace."
        "You MUST output **ONLY** a JSON object with 'tool_name' and 'tool_args'. "
        "If no tool is clearly identified, output 'UNKNOWN' for tool_name. "
        "Do NOT include any conversational text, explanations, or other text outside the JSON object."
        "\n\nExamples: "
        "User: 'Scale my 'my-app' deployment to 3 replicas in 'dev' namespace.'"
        "Output: {{\"tool_name\": \"scale_deployment\", \"tool_args\": {{\"deployment_name\": \"my-app\", \"replicas\": 3, \"namespace\": \"dev\"}}}}"
        "\nUser: 'List pods in production namespace.'"
        "Output: {{\"tool_name\": \"get_pods\", \"tool_args\": {{\"namespace\": \"production\"}}}}"
        "\nUser: 'What are the current pods?'"
        "Output: {{\"tool_name\": \"get_pods\", \"tool_args\": {{}}}}"
        "\nUser: 'Check the status of the cluster.'"
        "Output: {{\"tool_name\": \"UNKNOWN\", \"tool_args\": {{}}}}"
        "\n{format_instructions}"
    )),
    HumanMessage(content="{input}"),
])

async def extract_k8s_tool_call(user_msg: str) -> Optional[K8sToolCall]:
    """
    Uses an LLM to extract the K8s tool to call and its arguments from a user's query.
    """
    format_instructions = _k8s_tool_parser.get_format_instructions()
    messages = _K8S_EXTRACTION_PROMPT.format_messages(
        input=user_msg,
        format_instructions=format_instructions
    )
    llm_response = await asyncio.to_thread(_k8s_extraction_llm.invoke, messages)
    raw_response = llm_response.content.strip() if hasattr(llm_response, 'content') else str(llm_response).strip()

    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            json_string = match.group(0)
            parsed_call = _k8s_tool_parser.parse(json_string)
            if parsed_call.tool_name.upper() == "UNKNOWN":
                return None
            return parsed_call
        else:
            warnings.warn(f"LLM did not output valid JSON for K8s tool extraction. Raw: '{raw_response}'")
            return None
    except (ValidationError, json.JSONDecodeError) as e:
        warnings.warn(f"Failed to parse K8s tool extraction JSON: {e}. Raw: '{raw_response}'")
        return None

# ─── Helper for synthesizing final K8s answer ───
_k8s_synthesis_llm = get_llm(temperature=0.2, streaming=True) # Enable streaming for final output

async def synthesize_k8s_answer(user_query: str, tool_name: str, tool_output: Any) -> str:
    """
    Synthesizes a concise, human-readable answer for K8s queries.
    """
    final_synth_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful OpenShift Kubernetes assistant. Based on the following user query and tool output, "
            "provide a concise and direct answer to the user. "
            "If the tool output indicates success, confirm the action taken. If it indicates an error or no data, state that gracefully. "
            "DO NOT include any conversational filler, preambles, or additional questions. "
            "Just the answer."
        )),
        HumanMessage(content=user_query),
        AIMessage(content=f"Tool '{tool_name}' Output: {tool_output}"),
    ])
    
    final_synth_response = await asyncio.to_thread(_k8s_synthesis_llm.invoke, final_synth_prompt.format_messages(input=user_query, tool_name=tool_name, tool_output=tool_output))
    return final_synth_response.content.strip() if hasattr(final_synth_response, 'content') else str(final_synth_response).strip()

