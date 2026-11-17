"""``POST /chat``: answer a conversation by routing to a few tools and running a bounded loop.

``backend.main`` registers ``chat`` below at ``POST /chat``. The parts live in
``backend.helpers``: ``router`` (tool selection), ``orchestrator`` (the loop),
``executor`` (tool calls), ``model_adapter`` (OpenAI), ``prompts``, ``catalog``,
``config``, and ``contracts``.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from functools import lru_cache

from backend.chat_schema import ChatMessage, ChatRequest, ChatResponse, ToolCallLog
from backend.helpers.config import Settings, get_settings
from backend.helpers.contracts import ModelError
from backend.helpers.model_adapter import OpenAIResponsesAdapter
from backend.helpers.orchestrator import Orchestrator, history_from_messages
from backend.helpers.prompts import agent_instructions, search_plan
from backend.helpers.router import ToolRouter

logger = logging.getLogger("orchestrator")


def _configure_logging() -> None:
    """Send the "orchestrator" logger to stderr at INFO, once.

    Uvicorn configures only its own loggers, so without this the per-turn and
    per-tool lines would be dropped.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [orchestrator] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@lru_cache(maxsize=1)
def _components() -> tuple[Settings, ToolRouter, Orchestrator]:
    """Build the OpenAI client, router, and orchestrator once, on the first request.

    Lazy construction lets the server start (and ``/health`` answer) before a key
    is checked. One client is shared, so HTTP connections are reused.

    Returns:
        ``(settings, router, orchestrator)``.
    """
    from openai import OpenAI

    settings = get_settings()  # also loads .env when present
    # One SDK retry covers a transient 429/5xx without doubling the worst-case latency.
    client = OpenAI(timeout=settings.request_timeout_s, max_retries=1)
    router = ToolRouter(OpenAIResponsesAdapter(client, settings.router_model), settings)
    orchestrator = Orchestrator(
        OpenAIResponsesAdapter(client, settings.agent_model), settings, tool_loader=router.expand
    )
    return settings, router, orchestrator


def _current_time() -> datetime:
    """The mock workspace's clock, so "next week" matches the fixture data.

    Reads ``BASE_NOW`` from the calendar mock instead of calling the
    ``GOOGLECALENDAR_GET_CURRENT_DATE_TIME`` tool, which would add a logged call
    to every turn. Falls back to the system clock if the import fails.
    """
    try:
        from backend.googlecalendar_mock.state import BASE_NOW

        return BASE_NOW
    except Exception:
        return datetime.now().astimezone()


def chat(request: ChatRequest) -> ChatResponse:
    """Handle ``POST /chat``: route the latest request to a few tools, then run the loop.

    1. Router: one structured LLM call over a compact index of all 191 tools
       picks the few tools this turn may need (helpers/router.py).
    2. Orchestrator: the agent sees only those schemas and loops until it gives
       a final answer or hits a limit (helpers/orchestrator.py).
    3. Every tool invocation is logged in order in ``tool_calls``.

    Args:
        request: ``{"messages": [{"role", "content"}, ...]}``, the full
            conversation so far, oldest first.

    Returns:
        ``ChatResponse`` with the request messages plus one assistant message,
        and the ordered ``tool_calls`` log (name, arguments, result, error).
        Always HTTP 200: model failures produce an honest assistant message.
    """
    _configure_logging()
    started = time.perf_counter()
    _, router, orchestrator = _components()

    history = history_from_messages([m.model_dump() for m in request.messages])
    try:
        route = router.route(history)
    except ModelError as exc:  # fatal only: bad key, no quota, unknown model
        logger.error("router failed fatally: %s", exc)
        return _reply(request, "Sorry, I cannot answer right now because the language model is unavailable.", [])
    instructions = agent_instructions(_current_time())
    if route.search_all:  # unscoped request over an item that several services hold
        instructions += search_plan(route.search_all)
    try:
        result = orchestrator.run(instructions=instructions, history=history, tool_names=route.tools)
    except Exception:  # a bug in our code: answer honestly with HTTP 200, never a 500
        logger.exception("orchestrator crashed")
        return _reply(request, "Sorry, an internal error stopped this request before it finished.", [])

    logger.info(
        "turn done: route=%s %s dropped=%s | router %.0f ms tokens=%s | stop=%s steps=%d calls=%d | "
        "agent tokens=%s | total %.0f ms",
        route.source,
        route.tools,
        route.dropped,
        route.latency_ms,
        route.usage,
        result.stop_reason,
        result.steps,
        len(result.tool_calls),
        result.usage,
        (time.perf_counter() - started) * 1000,
    )
    return _reply(request, result.answer, result.tool_calls)


def _reply(request: ChatRequest, answer: str, tool_calls: list[ToolCallLog]) -> ChatResponse:
    """Append the assistant answer to the request messages and attach the tool log."""
    return ChatResponse(
        messages=[*request.messages, ChatMessage(role="assistant", content=answer)],
        tool_calls=tool_calls,
    )
