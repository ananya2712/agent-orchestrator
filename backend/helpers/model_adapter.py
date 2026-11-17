"""OpenAI Responses API adapter. Provider objects stop at this boundary.

External service: ``POST https://api.openai.com/v1/responses`` through the
official ``openai`` SDK (``client.responses.create``). Every call sends the full
conversation (``store=False``), so OpenAI keeps no state between calls.

``backend.main.get_openai_tools`` already emits Responses-format strict tool
schemas (flat ``name``/``parameters``), so the Responses API needs no conversion.
"""

from __future__ import annotations

import json
from typing import Any

import openai
from backend.helpers.contracts import UNPARSEABLE_ARGUMENTS, ModelError, ModelTurn, ToolCall


def _is_fatal(exc: openai.APIError) -> bool:
    """Return True when retrying the same request cannot succeed.

    Fatal: bad key (401), no permission (403), unknown model (404), and a 429
    caused by an empty balance. A plain rate limit (429) or a server error is
    transient, so it is not fatal.
    """
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        code = getattr(exc, "code", None) or ""
        return code in ("insufficient_quota", "credit_balance_exhausted")
    return False


def to_responses_input(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert provider-neutral conversation items into Responses API input items.

    Mapping:
        ``tool_call``   -> ``function_call`` (arguments re-encoded as a JSON string)
        ``tool_result`` -> ``function_call_output``
        ``user`` / ``assistant`` message -> the same role
        ``system`` message -> ``developer`` (the Responses API role for
        mid-conversation instructions)

    Args:
        items: Conversation items, oldest first.

    Returns:
        The ``input`` list for ``client.responses.create``.
    """
    converted: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type")
        if kind == "tool_call":
            converted.append(
                {
                    "type": "function_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": json.dumps(item["arguments"], default=str),
                }
            )
        elif kind == "tool_result":
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": item["output"],
                }
            )
        else:
            role = item["role"] if item["role"] in ("user", "assistant") else "developer"
            converted.append({"role": role, "content": item["content"]})
    return converted


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Decode a function call's JSON arguments. Bad JSON never raises here."""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        # Strict mode makes this rare; the executor refuses it as malformed_arguments.
        return {UNPARSEABLE_ARGUMENTS: raw}
    return parsed if isinstance(parsed, dict) else {UNPARSEABLE_ARGUMENTS: raw}


def _usage(response: Any) -> dict[str, int]:
    """Extract input, output, and cached token counts for the turn log."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "cached_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
    }


def _supports_temperature(model: str) -> bool:
    """False for reasoning models (o-series, gpt-5), which reject ``temperature``."""
    return not model.startswith(("o1", "o3", "o4", "gpt-5"))


class OpenAIResponsesAdapter:
    """``ModelAdapter`` implementation backed by the OpenAI Responses API."""

    def __init__(self, client: Any, model: str) -> None:
        """
        Args:
            client: An ``openai.OpenAI`` client (or a test double with the same
                ``responses.create`` method).
            model: Model ID, for example ``gpt-4.1-mini``.
        """
        self.client = client
        self.model = model
        # Token usage of the latest complete_json call; the router reports it in
        # the turn log (complete() returns usage on the ModelTurn instead).
        self.last_usage: dict[str, int] = {}

    def _base_kwargs(self, instructions: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Request options shared by both call types: model, prompt, input, no storage."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": to_responses_input(items),
            "store": False,
        }
        if _supports_temperature(self.model):
            kwargs["temperature"] = 0  # repeatable tool choices across runs
        return kwargs

    def _create(self, **kwargs: Any) -> Any:
        """Call the API; turn every SDK error into a ``ModelError`` with a fatal flag."""
        try:
            return self.client.responses.create(**kwargs)
        except openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}", fatal=_is_fatal(exc)) from exc

    def complete(
        self,
        *,
        instructions: str,
        items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ModelTurn:
        """Run one agent step. See ``ModelAdapter.complete`` for the contract.

        ``parallel_tool_calls=True`` lets the model request independent calls in
        one step, which saves a round trip. The loop still runs them in order.
        """
        kwargs = self._base_kwargs(instructions, items)
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice, parallel_tool_calls=True)
        response = self._create(**kwargs)

        # A response can mix text and function calls; keep both.
        calls = tuple(
            ToolCall(id=item.call_id, name=item.name, arguments=_parse_arguments(item.arguments))
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        )
        text = (response.output_text or "").strip() or None
        return ModelTurn(content=text, tool_calls=calls, usage=_usage(response))

    def complete_json(
        self,
        *,
        instructions: str,
        items: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run one structured-output call. See ``ModelAdapter.complete_json``.

        Uses ``text.format = json_schema`` with ``strict=True``, so the reply is
        guaranteed to parse and match the schema.
        """
        kwargs = self._base_kwargs(instructions, items)
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens
        kwargs["text"] = {
            "format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}
        }
        response = self._create(**kwargs)
        self.last_usage = _usage(response)
        return json.loads(response.output_text)
