"""
PlannerAgent
============
Generates a JSON task list for the graph.  (Unchanged logic—only the import
statement below is corrected to match the real filename.)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

# ← FIX: point at the *existing* base_agent.py
from .base_agent import BaseAgent

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helpers (unchanged)
# --------------------------------------------------------------------------- #

SMART_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}
_JSON_RE = re.compile(r"```json\s*(.*?)```|(\{.*?\})", re.DOTALL)


def _normalize_quotes(text: str) -> str:
    return text.translate(str.maketrans(SMART_QUOTES))


def _extract_json(text: str) -> str | None:
    m = _JSON_RE.search(text)
    return m.group(1) or m.group(2) if m else None


def _safe_parse_tasks(raw: str) -> List[Dict[str, Any]]:
    cleaned = _normalize_quotes(raw)
    block = _extract_json(cleaned)
    if not block:
        LOGGER.warning("PlannerAgent: no JSON block found.")
        return []

    try:
        data = json.loads(block)
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            LOGGER.warning("PlannerAgent: 'tasks' is not a list.")
            return []
        return tasks
    except json.JSONDecodeError as err:
        LOGGER.error("PlannerAgent: JSON error – %s", err, exc_info=True)
        return []


# --------------------------------------------------------------------------- #
# PlannerAgent class (unchanged)
# --------------------------------------------------------------------------- #


class PlannerAgent(BaseAgent):  # noqa: D101
    name = "planner"

    @staticmethod
    def default_fallback() -> str:  # noqa: D401
        return (
            "⚠️  PlannerAgent couldn’t decide which tool to run. "
            "Check the planner logs for JSON-parsing errors."
        )

    @staticmethod
    def _build_prompt(summary: str, user_msg: str) -> str:
        return (
            "You are the PlannerAgent inside an agentic AI system.\n"
            "Return STRICT JSON with a single array field `tasks`.\n\n"
            "Example:\n"
            "```json\n"
            '{"tasks":[{"id":"finance"}]}\n'
            "```\n\n"
            f"--- Conversation summary (≤250 tokens) ---\n{summary}\n"
            f"--- Latest user message ---\n{user_msg}\n"
            "--- End ---"
        )

    # public API ------------------------------------------------------------ #

    def plan(  # noqa: D401
        self,
        conversation_summary: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(conversation_summary, user_message)

        llm_resp = self.llm.invoke(
            prompt,
            model=model or self.default_model,
            temperature=temperature,
            response_format={"type": "json_object"},  # OpenAI / Anthropic
        )

        raw_text: str = (
            llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)
        )
        LOGGER.debug("PlannerAgent raw output:\n%s", raw_text)

        return _safe_parse_tasks(raw_text)
