"""Provider-neutral data types shared by the router, loop, and executor.

Conversation items are plain dicts with one of three shapes:

* ``{"role": "user" | "assistant" | "system", "content": str}``
* ``{"type": "tool_call", "call_id": str, "name": str, "arguments": dict}``
* ``{"type": "tool_result", "call_id": str, "output": str}``

Only ``model_adapter`` knows how these map onto the OpenAI wire format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model.

    Attributes:
        id: Provider call ID. The matching ``tool_result`` must reuse it.
        name: Tool name as the model wrote it (may be unknown).
        arguments: Parsed JSON arguments, not yet validated.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    """One model response: final text, tool calls, or (invalid) neither."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)


TOOL_FAILED_HINT = (
    "If you passed an ID, retry once with the display name (or the reverse). "
    "If a needed value is missing, look it up with a tool before asking the user."
)


# Key the adapter uses when the model's arguments are not a JSON object.
UNPARSEABLE_ARGUMENTS = "_unparseable_arguments"


def is_empty_listing(value: Any) -> bool:
    """True for ``[]`` or a dict whose list fields are all empty (``{"users": [], "count": 0}``)."""
    if isinstance(value, list):
        return not value
    if not isinstance(value, dict):
        return False
    lists = [v for v in value.values() if isinstance(v, list)]
    return bool(lists) and all(not v for v in lists)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call: a value on success, or an error code and message.

    Error codes: ``unknown_tool``, ``invalid_arguments``, ``tool_failed`` (from the
    executor) and ``repeated_tool_call``, ``tool_call_limit``, ``tool_request_limit``
    (from the loop).
    """

    call_id: str
    name: str
    ok: bool
    value: Any = None
    error_code: str | None = None
    error_message: str | None = None
    hint: str | None = None  # extra guidance for the model only (never logged)

    def observation(self, char_limit: int) -> str:
        """Serialize this result as the text the model sees on its next step.

        Args:
            char_limit: Longest text to return. Longer text is cut and marked, so
                one large result cannot flood the context. The full value stays
                in the ``tool_calls`` log.

        Returns:
            Compact JSON: ``{"result": ...}`` or ``{"error": {"code", "message"}}``.
        """
        if self.ok:
            payload: dict[str, Any] = {"result": self.value}
            if self.hint:
                payload["hint"] = self.hint
        else:
            payload = {
                "error": {"code": self.error_code, "message": self.error_message}
            }
            if self.error_code == "tool_failed":
                # Recovery hint at the moment it matters; the log keeps the bare error.
                payload["error"]["hint"] = TOOL_FAILED_HINT
        text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        if len(text) <= char_limit:
            return text
        dropped = len(text) - char_limit
        return f"{text[:char_limit]}...[truncated {dropped} chars; narrow the query for more]"


class ModelError(Exception):
    """A provider call failed. ``fatal`` means retrying this request cannot help
    (bad key, no quota, unknown model), so callers should stop at once."""

    def __init__(self, message: str, *, fatal: bool) -> None:
        """
        Args:
            message: Error text for the log (never shown to the user verbatim).
            fatal: True when retrying the same request cannot succeed.
        """
        super().__init__(message)
        self.fatal = fatal


class ModelAdapter(Protocol):
    """What the router and the loop need from a model provider.

    ``OpenAIResponsesAdapter`` is the real implementation. Tests use a scripted
    fake, so the loop can be checked without network calls.
    """

    def complete(
        self,
        *,
        instructions: str,
        items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ModelTurn:
        """Run one model step that may call tools.

        Args:
            instructions: System prompt for this step.
            items: Conversation items (see the module docstring).
            tools: Tool schemas the model may call. May be empty.
            tool_choice: ``"auto"``, or ``"none"`` to force a text answer.

        Returns:
            The model's text, its tool calls, and token usage.

        Raises:
            ModelError: The provider call failed.
        """
        ...

    def complete_json(
        self,
        *,
        instructions: str,
        items: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run one model call whose output must match a JSON schema.

        Args:
            instructions: System prompt.
            items: Conversation items.
            schema: Strict JSON schema for the output.
            schema_name: Name the provider shows for the schema.
            max_output_tokens: Optional cap on output length.

        Returns:
            The parsed JSON object.

        Raises:
            ModelError: The provider call failed.
        """
        ...
