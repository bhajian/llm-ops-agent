"""
PlannerAgent
============
Generates an ordered list of task IDs in JSON.
If the LLM returns malformed JSON, we gracefully return an empty list,
allowing graph_builder to choose a safe default behaviour.
"""

import json
import re
from typing import List

from ..llm import get_llm
from ..integrations.tools import MemorySummaryTool

_TASK_IDS = {"retrieval", "finance", "k8s"}

# ────────────────── helpers ──────────────────
def _safe_parse_tasks(text: str) -> List[str]:
    """
    Extract first {...} JSON block and return validated task list.
    On any failure, return []  (caller will decide fallback).
    """
    try:
        # Step 1: Aggressively clean the input text by stripping non-ASCII characters.
        # This helps in case there are invisible characters that disrupt JSON parsing.
        cleaned_text_for_parsing = text.encode('ascii', 'ignore').decode('ascii')

        # Step 2: Use regex to find the JSON block. This will prioritize 'json' code blocks
        # or fall back to finding any {...} block.
        block_match = re.search(r"```json\n(.*?)```|(\{.*?\})", cleaned_text_for_parsing, re.DOTALL | re.S)
        
        if block_match:
            # Extract the JSON string from the matched group, prioritizing the markdown block.
            # Then, strip any leading/trailing whitespace.
            json_str = (block_match.group(1) or block_match.group(2)).strip()
            data = json.loads(json_str)
        else:
            raise ValueError("No JSON found in text.")

        # Step 3: Extract and validate tasks.
        tasks = [
            t["id"] if isinstance(t, dict) else t
            for t in data.get("tasks", [])
        ]
        return [tid for tid in tasks if tid in _TASK_IDS]
    except Exception as err:
        # Log the raw text that caused the parsing failure for debugging.
        print(f"⚠️ Planner JSON parse failed: {err}. Raw text: {text}")
        return []


# ────────────────── main entry ──────────────────
def run(user_query: str, conversation_id: str) -> List[str]:
    summary_tool = MemorySummaryTool()
    summary = summary_tool.run(conversation_id)

    llm = get_llm(streaming=False, temperature=0.0)
    # Refined prompt for clearer instructions to the LLM,
    # especially for greetings and when no specific tool is needed.
    prompt = (
        "You are PlannerAgent. Your role is to determine which domain-specific agents "
        "should be executed based on the user's query and the conversation summary.\n"
        "Allowed agent IDs are: retrieval, finance, k8s.\n"
        "You MUST return a valid JSON object with a single top-level key named 'tasks'.\n"
        "The value of 'tasks' MUST be an array of objects, where each object has an 'id' key "
        "corresponding to an allowed agent ID.\n"
        "Example for a single 'finance' task: {\"tasks\":[{\"id\":\"finance\"}]}\n"
        "Example for multiple tasks: {\"tasks\":[{\"id\":\"retrieval\"},{\"id\":\"k8s\"}]}\n"
        "Specific example: If the user asks about a stock price (e.g., 'What is the price of AAPL?'), "
        "you MUST return: {\"tasks\":[{\"id\":\"finance\"}]}\n"
        "If the user's query is a simple greeting (e.g., 'Hi', 'Hello'), or does not require a specific tool, "
        "return an empty array for tasks: {\"tasks\":[]}\n" # Added specific instruction for greetings/no-tool queries
        "DO NOT include any other text, explanation, or markdown outside of the JSON.\n"
        "Only respond with the JSON object.\n\n"
        f"Conversation summary:\n{summary}\n\n"
        f"User's latest question: {user_query}"
    )

    llm_response = llm.invoke(prompt).content
    return _safe_parse_tasks(llm_response)
