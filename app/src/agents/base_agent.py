"""
BaseAgent
=========
Common functionality shared by all concrete agents.

⚠️  If you already had a full-featured BaseAgent earlier, paste your
    original implementation here and delete the minimal example below.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Minimal reference implementation — extend as needed
# --------------------------------------------------------------------------- #

class BaseAgent:  # noqa: D101
    """
    Very small base class.

    * `llm`      – anything with an `.invoke(prompt, **kwargs)` method
    * `name`     – override in subclasses
    * `default_model` – override if your LLM client needs it
    """

    name: str = "base"
    default_model: str = "gpt-4o-mini"

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, llm: Any) -> None:  # noqa: ANN401 (generic `Any` is OK)
        self.llm = llm

    # ------------------------------------------------------------------ #
    # Utility helpers that children commonly use
    # ------------------------------------------------------------------ #

    def call_llm(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        **extra,
    ):
        """
        Thin wrapper around whatever LLM client you pass in.
        Children can call `self.call_llm(...)` instead of touching
        `self.llm.invoke(...)` directly.
        """
        return self.llm.invoke(
            prompt,
            model=model or self.default_model,
            temperature=temperature,
            **extra,
        )

