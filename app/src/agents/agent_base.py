"""
agent_base.py
-------------
Re-export the real BaseAgent so legacy imports work:

    from .agent_base import BaseAgent
"""
from .base_agent import BaseAgent  # noqa: F401

__all__: list[str] = ["BaseAgent"]
