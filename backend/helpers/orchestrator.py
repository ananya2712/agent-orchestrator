"""The bounded agent loop.

Each step: ask the model -> final text (stop) or tool calls -> handle the calls
in request order -> feed the results back. Hard limits on steps and tool calls;
an identical repeated call does not run twice. When the loop stops early, one
last model call with tool_choice="none" writes an honest answer.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.chat_schema import ToolCallLog
from backend.helpers import catalog
from backend.helpers.config import Settings
from backend.helpers.contracts import ModelAdapter, ModelError, ModelTurn, ToolCall, ToolResult
from backend.helpers.executor import execute
from backend.helpers.prompts import FINAL_ANSWER_NUDGE

logger = logging.getLogger("orchestrator")

# A meta-tool, not a registered tool: it changes which schemas the agent sees,
# runs no service code, and is therefore not written to the tool_calls log.
REQUEST_TOOLS = "request_tools"
REQUEST_TOOLS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": REQUEST_TOOLS,
    "description": (
        "Load more tools. Call this only when none of your current tools can do a step "
        "the user needs (for example, a service or action you have no tool for). "
        "Describe the missing capability in a few words."
    ),
    "parameters": {
        "type": "object",
        "properties": {"need": {"type": "string", "description": "The missing capability."}},
        "required": ["need"],
        "additionalProperties": False,
    },
    "strict": True,
}
MAX_TOOL_REQUESTS = 2
# Steps in a row where every call was refused before the loop stops as loop_detected.
LOOP_STREAK_LIMIT = 2
# Refusals that do not count as a tool's failure (the call was fine, just redundant).
_NOT_A_FAILURE = {"repeated_tool_call", "tool_call_limit", "tool_retry_limit"}

# (need, currently active tool names) -> names to add
ToolLoader = Callable[[str, list[str]], list[str]]


@dataclass
class RunResult:
    """What one turn produced.

    Attributes:
        answer: Final assistant text. Never empty.
        tool_calls: Ordered log of every invoked tool, for the response.
        stop_reason: Why the loop ended (see the comment on the field).
        steps: Model calls made inside the loop.
        usage: Summed token counts across the turn's model calls.
    """

    answer: str
    tool_calls: list[ToolCallLog]
    stop_reason: str  # complete | step_limit | tool_call_limit | loop_detected | deadline | invalid_model_response | model_error
    steps: int = 0
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class _RunState:
    """Everything one /chat turn mutates, kept in one place so the loop stays flat."""

    items: list[dict[str, Any]]
    active_tools: list[str]
    schemas: list[dict[str, Any]] = field(default_factory=list)
    logs: list[ToolCallLog] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    usage: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    tool_requests: int = 0
    failures: dict[str, int] = field(default_factory=dict)  # failed or refused calls per tool name
    all_refused_streak: int = 0  # consecutive steps in which every call was refused
    stop_reason: str = "step_limit"

    def add_usage(self, turn: ModelTurn) -> None:
        """Add one model call's token counts to the turn total."""
        for key, value in turn.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value

    def finish(self, answer: str, stop_reason: str | None = None) -> RunResult:
        """Freeze the state into a ``RunResult``; ``stop_reason`` overrides the stored one."""
        return RunResult(answer, self.logs, stop_reason or self.stop_reason, self.steps, self.usage)


def history_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map the /chat wire messages onto conversation items.

    The wire format has no call IDs, so an incoming ``tool`` message cannot be
    replayed as a function output. It is kept as assistant-side context instead.

    Args:
        messages: ``ChatMessage`` dicts from the request, oldest first.

    Returns:
        Conversation items for the router and the loop.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "tool":
            items.append({"role": "assistant", "content": f"[Earlier tool output]\n{content}"})
        else:
            items.append({"role": role, "content": content})
    return items


def _fingerprint(call: ToolCall) -> str:
    """Stable key for "same tool, same arguments" (key order does not matter)."""
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


class Orchestrator:
    """Runs the bounded tool-calling loop for one /chat turn."""

    def __init__(
        self, model: ModelAdapter, settings: Settings, tool_loader: ToolLoader | None = None
    ) -> None:
        """
        Args:
            model: Adapter for the agent model.
            settings: Step, call, and observation limits.
            tool_loader: Backs the ``request_tools`` meta-tool. ``None`` hides the
                meta-tool (used in some tests).
        """
        self.model = model
        self.settings = settings
        self.tool_loader = tool_loader

    def run(
        self, *, instructions: str, history: list[dict[str, Any]], tool_names: list[str]
    ) -> RunResult:
        """Run the loop until the model answers or a limit stops it.

        Args:
            instructions: Agent system prompt.
            history: Conversation items from ``history_from_messages``.
            tool_names: Tools chosen by the router; their schemas are all the
                agent sees (plus ``request_tools`` when a loader is set).

        Returns:
            The answer, the ordered tool log, the stop reason, and usage. This
            method does not raise: model failures become ``stop_reason="model_error"``.
        """
        state = _RunState(items=list(history), active_tools=list(tool_names))
        state.schemas = self._schemas(state.active_tools)
        deadline = time.monotonic() + self.settings.turn_deadline_s

        while state.steps < self.settings.max_steps:
            if time.monotonic() > deadline:
                # No more model calls: report what ran, honestly, without waiting again.
                logger.error("turn deadline of %.0f s reached at step %d", self.settings.turn_deadline_s, state.steps)
                return state.finish(_fallback_answer(state.logs), "deadline")
            # 1. Ask the model for the next step.
            try:
                turn = self.model.complete(instructions=instructions, items=state.items, tools=state.schemas)
            except Exception as exc:
                if isinstance(exc, ModelError):
                    logger.error("model call failed at step %d: %s", state.steps + 1, exc)
                else:
                    logger.exception("model call failed at step %d", state.steps + 1)
                return state.finish(_model_error_answer(state.logs, exc), "model_error")
            state.steps += 1
            state.add_usage(turn)

            # 2. No tool calls: text is the final answer; nothing at all is invalid.
            if not turn.tool_calls:
                if turn.content:
                    return state.finish(turn.content, "complete")
                state.stop_reason = "invalid_model_response"
                break

            # 3. Tool calls. Every function_call needs a matching output, so record
            #    all calls first, then answer each one (executed or refused).
            if turn.content:
                # Text next to tool calls is the model's working note (for example a
                # condition check); keep it in context and in the log, not in the reply.
                logger.info("step %d model note: %s", state.steps, turn.content[:500].replace("\n", " | "))
                state.items.append({"role": "assistant", "content": turn.content})
            state.items += [
                {"type": "tool_call", "call_id": c.id, "name": c.name, "arguments": c.arguments}
                for c in turn.tool_calls
            ]
            refused = 0
            for call in turn.tool_calls:
                result = self._handle(call, state)
                refused += result.error_code is not None and result.error_code not in ("tool_failed", "invalid_arguments", "tool_timeout")
                state.items.append(
                    {
                        "type": "tool_result",
                        "call_id": call.id,
                        "output": result.observation(self.settings.observation_char_limit),
                    }
                )
            if state.stop_reason == "tool_call_limit":
                break
            # A model that only repeats refused calls is stuck; stop early.
            state.all_refused_streak = state.all_refused_streak + 1 if refused == len(turn.tool_calls) else 0
            if state.all_refused_streak >= LOOP_STREAK_LIMIT:
                state.stop_reason = "loop_detected"
                break

        # 4. Early stop (step limit, call limit, empty turn): one tool-free call
        #    writes an honest answer from the results so far.
        answer = self._final_answer(instructions, state)
        return state.finish(answer or _fallback_answer(state.logs))

    def _handle(self, call: ToolCall, state: _RunState) -> ToolResult:
        """Decide what happens to one call, in this order:

        1. ``request_tools`` -> load more tools (not logged).
        2. A registered tool the agent was not given -> refuse (``tool_not_loaded``).
        3. Call limit reached -> refuse, and mark the turn to stop.
        4. This tool already failed ``max_failures_per_tool`` times -> refuse.
        5. Same tool and arguments already ran -> refuse.
        6. Otherwise -> execute (with a timeout); log it if it ran; count a failure
           for any error except a redundant-call refusal.
        """
        if call.name == REQUEST_TOOLS and self.tool_loader is not None:
            return self._load_tools(call, state)
        if catalog.has_tool(call.name) and call.name not in state.active_tools:
            return _refusal(
                call, "tool_not_loaded",
                f"{call.name} is not one of your tools for this request. Call request_tools to load it first.",
            )
        if len(state.logs) >= self.settings.max_tool_calls:
            state.stop_reason = "tool_call_limit"
            return _refusal(call, "tool_call_limit", "Tool call limit reached for this request.")
        if state.failures.get(call.name, 0) >= self.settings.max_failures_per_tool:
            return _refusal(
                call, "tool_retry_limit",
                f"{call.name} already failed {state.failures[call.name]} times in this request. Stop retrying it; "
                "answer with what you have and say what could not be done.",
            )
        fingerprint = _fingerprint(call)
        if fingerprint in state.seen:
            return _refusal(
                call, "repeated_tool_call", "This exact call already ran in this request. Use its earlier result."
            )
        state.seen.add(fingerprint)

        started = time.perf_counter()
        result, log = execute(call, timeout_s=self.settings.tool_timeout_s)
        if log is not None:
            state.logs.append(log)
        if not result.ok and result.error_code not in _NOT_A_FAILURE:
            state.failures[call.name] = state.failures.get(call.name, 0) + 1
        logger.info(
            "step %d tool %s %s (%.0f ms)",
            state.steps,
            call.name,
            "ok" if result.ok else result.error_code,
            (time.perf_counter() - started) * 1000,
        )
        return result

    def _schemas(self, active: list[str]) -> list[dict[str, Any]]:
        """Schemas for the active tools, plus the meta-tool when a loader exists."""
        schemas = catalog.openai_schemas(active)
        if self.tool_loader is not None:
            schemas.append(REQUEST_TOOLS_SCHEMA)
        return schemas

    def _load_tools(self, call: ToolCall, state: _RunState) -> ToolResult:
        """Handle ``request_tools``: add the loader's tools to the next model call.

        Returns a success result that lists the added names. If nothing matched,
        the result tells the model to report that the task cannot be done.
        """
        state.tool_requests += 1
        if state.tool_requests > MAX_TOOL_REQUESTS:
            return _refusal(call, "tool_request_limit", "No more tools can be loaded. Answer with what you have.")
        need = str(call.arguments.get("need", "")).strip()
        added = self.tool_loader(need, list(state.active_tools)) if need and self.tool_loader else []
        logger.info("request_tools %r -> %s", need, added)
        if not added:
            note = "No matching tool exists. Tell the user this cannot be done."
            return ToolResult(call.id, call.name, ok=True, value={"added_tools": [], "note": note})
        state.active_tools += added
        state.schemas = self._schemas(state.active_tools)
        return ToolResult(call.id, call.name, ok=True, value={"added_tools": added})

    def _final_answer(self, instructions: str, state: _RunState) -> str | None:
        """Make one last call with ``tool_choice="none"``. Returns ``None`` if it fails."""
        nudge = {"role": "system", "content": FINAL_ANSWER_NUDGE}
        try:
            turn = self.model.complete(
                instructions=instructions, items=[*state.items, nudge], tools=state.schemas, tool_choice="none"
            )
        except Exception as exc:
            logger.error("final-answer call failed: %s", exc)
            return None
        state.add_usage(turn)
        return turn.content


def _refusal(call: ToolCall, code: str, message: str) -> ToolResult:
    """Error result for a call the loop declined to run."""
    return ToolResult(call.id, call.name, ok=False, error_code=code, error_message=message)


def _summary(logs: list[ToolCallLog]) -> str:
    """One sentence naming the tools that ran, for the fixed fallback answers."""
    if not logs:
        return "No tools were run."
    return "Tools run before stopping: " + ", ".join(log.name for log in logs) + "."


def _model_error_answer(logs: list[ToolCallLog], exc: Exception) -> str:
    """Answer used when a model call fails mid-loop; names the error type only."""
    return (
        "Sorry, I could not finish this request because the language model call failed "
        f"({type(exc).__name__}). {_summary(logs)}"
    )


def _fallback_answer(logs: list[ToolCallLog]) -> str:
    """Answer used when even the forced final call returns no text."""
    return f"Sorry, I could not complete this request. {_summary(logs)}"
