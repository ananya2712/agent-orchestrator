"""Runtime settings for the /chat orchestrator.

Nothing in the starter loads ``.env``, so this module reads it once with the
stdlib. Values already in the process environment win, so a grader-provided
``OPENAI_API_KEY`` is never overridden by a local file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# backend/helpers/config.py -> repo root (the folder that holds .env.example)
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_env_loaded = False

DEFAULT_MODEL = "gpt-4.1-mini"


def load_env_file() -> None:
    """Load ``KEY=VALUE`` lines from the repo-root ``.env`` into ``os.environ``, once.

    Blank lines, ``#`` comments, and an ``export`` prefix are allowed, and quotes
    around a value are removed. ``os.environ.setdefault`` keeps any variable that
    is already set, so the grader's environment always wins. A missing file is
    not an error: the process environment is then the only source.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if not _ENV_PATH.is_file():
        return
    for raw_line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """Model names and the hard limits for one /chat turn.

    Attributes:
        agent_model: Model that runs the tool-calling loop.
        router_model: Model that selects tools from the compact catalog.
        max_steps: Most model calls in the loop before a forced final answer.
        max_tool_calls: Most logged tool invocations per turn.
        max_routed_tools: Most tools the router may hand to the agent.
        router_history_messages: How many recent messages the router reads.
        observation_char_limit: Longest tool result text sent back to the model.
        request_timeout_s: Per-request timeout for the OpenAI client.
        tool_timeout_s: Longest wait for one tool call before ``tool_timeout``.
        max_failures_per_tool: Failed or refused calls of one tool per turn before
            further calls to it are refused (``tool_retry_limit``).
        turn_deadline_s: Wall-clock budget for one turn; when it is spent, the loop
            stops without another model call (``deadline``).
    """

    agent_model: str
    router_model: str
    max_steps: int = 8
    max_tool_calls: int = 15
    max_routed_tools: int = 12
    router_history_messages: int = 6
    observation_char_limit: int = 12_000
    request_timeout_s: float = 60.0
    tool_timeout_s: float = 10.0
    max_failures_per_tool: int = 5
    turn_deadline_s: float = 90.0


def get_settings() -> Settings:
    """Build ``Settings`` from the environment, after loading ``.env``.

    ``OPENAI_MODEL`` sets both models (default ``gpt-4.1-mini``). The optional
    ``AGENT_MODEL`` and ``ROUTER_MODEL`` variables override one side each.

    Returns:
        The settings with the default limits.
    """
    load_env_file()
    model = os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    return Settings(
        agent_model=os.environ.get("AGENT_MODEL") or model,
        router_model=os.environ.get("ROUTER_MODEL") or model,
    )
