"""Validates and runs one tool call. Never raises to the caller.

A call goes through two phases:

1. Pre-checks (the tool never runs, so nothing is logged in ``tool_calls``):

   ======================== ===================================================
   unknown_tool             the name is not in the registry
   malformed_arguments      the model's arguments were not valid JSON
   missing_required_value   a required field is null or blank
   invalid_query_syntax     invented ``field:value`` text in a plain-text search
   ======================== ===================================================

2. Invocation (logged in ``tool_calls``, with ``error`` set on failure):

   ======================== ===================================================
   invalid_arguments        the arguments fail the tool's Pydantic model
   tool_failed              the tool raised (for example, an unknown ID)
   tool_timeout             the tool did not return within the time limit
   ======================== ===================================================

A successful call that returns only empty lists gets a ``hint`` that names the
filters the model used, so it retries without the guessed ones.

Every refusal and failure carries a message that tells the model how to fix the
call, so it can recover instead of giving up or guessing.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from backend.chat_schema import ToolCallLog
from backend.helpers import catalog
from backend.helpers.contracts import UNPARSEABLE_ARGUMENTS, ToolCall, ToolResult, is_empty_listing
from backend.tooling import ToolSpec
from pydantic import ValidationError

logger = logging.getLogger("orchestrator")

# In these services every search parameter is a plain substring match (checked in
# the mock code), so "field:value" text can never match anything. Gmail and Slack
# are excluded because they do parse operators such as from: and in:.
PLAIN_TEXT_SEARCH_SERVICES = frozenset({"linear", "googlecalendar", "github", "googledrive"})
SEARCH_PARAMETERS = frozenset({"query", "q", "search", "filter"})
# A word followed by ":" and a non-space, non-slash character, at the start of the
# value or after a space or "(". Matches "project:abc" and "is:open"; skips times
# ("10:00"), URLs ("https://"), and prose ("Note: x").
_FIELD_VALUE_SYNTAX = re.compile(r"(?:^|[\s(])[A-Za-z_][\w.-]*:(?=[^\s/])")



class ToolTimeout(Exception):
    """The tool did not return within the time limit."""


def _invoke_with_timeout(spec: ToolSpec, arguments: dict[str, Any], timeout_s: float | None) -> Any:
    """Run ``spec.invoke`` on its own daemon thread and wait at most ``timeout_s``.

    One thread per call (not a fixed pool), so hung calls can never fill a pool and
    delay later ones. Python cannot kill a thread, so a timed-out call may finish
    in the background; the in-process mocks never hang, so this is a safety net.

    Raises:
        ToolTimeout: No result within ``timeout_s``.
        Exception: Whatever the tool raised.
    """
    if timeout_s is None:
        return spec.invoke(**arguments)
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = spec.invoke(**arguments)
        except Exception as exc:  # re-raised in the caller's thread below
            outcome["error"] = exc

    worker = threading.Thread(target=target, name=f"tool-{spec.name}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ToolTimeout
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


# --------------------------------------------------------------------------- public API
def execute(call: ToolCall, timeout_s: float | None = None) -> tuple[ToolResult, ToolCallLog | None]:
    """Run one tool call against the in-process mock registry.

    Args:
        call: The call as the model requested it.
        timeout_s: Longest wait for the tool; ``None`` waits without a limit.

    Returns:
        A pair: the ``ToolResult`` for the model, and the ``ToolCallLog`` entry
        for the response. The log entry is ``None`` when a pre-check refused the
        call, because the tool never ran.
    """
    # Phase 1: pre-checks. Order matters: a name must exist before its spec can be read.
    if not catalog.has_tool(call.name):
        return _refuse(call, "unknown_tool", f"No tool named {call.name!r}. Use only the tools you were given.")

    if UNPARSEABLE_ARGUMENTS in call.arguments:
        return _refuse(
            call, "malformed_arguments",
            "Your arguments were not valid JSON. Resend the call with a valid JSON object of arguments.",
        )

    spec = catalog.get_spec(call.name)
    arguments = clean_arguments(spec, call.arguments)

    missing = missing_required_values(spec, arguments)
    if missing:
        return _refuse(
            call,
            "missing_required_value",
            f"Required value(s) missing: {', '.join(missing)}. Do not guess or use placeholders. "
            "Find the value with another tool if one exists; otherwise ask the user for it.",
        )

    syntax_error = invented_query_syntax(spec.service, arguments)
    if syntax_error is not None:
        return _refuse(call, "invalid_query_syntax", syntax_error)

    # Phase 2: invocation. Validate first so bad arguments never reach the handler.
    try:
        spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        return _failure(call, arguments, "invalid_arguments", _validation_message(exc))

    # invoke() wraps the return value as {"result": ...}; keep only the inner
    # JSON-safe value, as in the README's tool_calls example.
    try:
        value = _invoke_with_timeout(spec, arguments, timeout_s).model_dump(mode="json")["result"]
    except ToolTimeout:
        return _failure(call, arguments, "tool_timeout", f"No result after {timeout_s:g} s. Do not retry this call.")
    except Exception as exc:  # mock errors are ValueError subclasses; catch all
        return _failure(call, arguments, "tool_failed", str(exc) or type(exc).__name__)

    hint = empty_result_hint(spec, arguments) if is_empty_listing(value) else None
    result = ToolResult(call_id=call.id, name=call.name, ok=True, value=value, hint=hint)
    return result, ToolCallLog(name=call.name, arguments=arguments, result=value, error=None)


# --------------------------------------------------------------------------- argument checks
def clean_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Drop nulls for optional fields.

    Strict mode makes the model send every field, using null for the ones it does
    not need. Dropping them lets the tool's own default apply, and keeps the log
    readable. A null for a required field is kept so the required-value check can
    report it.

    Args:
        spec: The tool being called.
        arguments: Arguments exactly as the model sent them.

    Returns:
        A new dict without the null optional fields. Unknown keys are kept, so
        validation can reject them.
    """
    fields = spec.args_model.model_fields
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        field = fields.get(key)
        if value is None and field is not None and not field.is_required():
            continue
        cleaned[key] = value
    return cleaned


def missing_required_values(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """Names of required fields whose value is absent, null, blank, or an empty list.

    Many mock tools declare required fields as nullable (``str | None``), so
    Pydantic accepts ``null`` and the tool itself fails. Checking first keeps a
    call that could never succeed out of the log, and tells the model to ask.
    """
    missing = []
    for name, field in spec.args_model.model_fields.items():
        if not field.is_required():
            continue
        value = arguments.get(name)
        if value is None or value == [] or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    return missing


def invented_query_syntax(service: str, arguments: dict[str, Any]) -> str | None:
    """Return a fix-it message if a plain-text search argument contains field:value syntax.

    Args:
        service: The tool's service name (``ToolSpec.service``).
        arguments: Cleaned call arguments.

    Returns:
        A message that tells the model how to fix the call, or ``None`` if the
        arguments are acceptable (always ``None`` for Gmail and Slack).
    """
    if service not in PLAIN_TEXT_SEARCH_SERVICES:
        return None
    for key, value in arguments.items():
        if key in SEARCH_PARAMETERS and isinstance(value, str) and _FIELD_VALUE_SYNTAX.search(value):
            return (
                f"{key}={value!r} uses field:value syntax, but {service} search is plain text, so it "
                "would match nothing. Put IDs in their own parameters (for example team=<id>) and "
                "use only plain words here, or send null."
            )
    return None


def empty_result_hint(spec: ToolSpec, arguments: dict[str, Any]) -> str:
    """Guidance for an empty listing: name the optional filters this call used.

    Args:
        spec: The tool that returned nothing.
        arguments: The cleaned arguments of that call.

    Returns:
        A retry instruction naming each optional argument whose value differs
        from the tool's default, or a "report nothing found" note when the call
        used no such filter (then the result really is empty).
    """
    fields = spec.args_model.model_fields
    filters = {
        key: value for key, value in arguments.items()
        if key in fields and not fields[key].is_required() and value != fields[key].default
        and key not in ("limit", "orderBy", "page", "perPage", "max_results", "pageSize")
    }
    if not filters:
        return "No items match. This call used no filters, so report that nothing was found."
    used = ", ".join(f"{key}={value!r}" for key, value in filters.items())
    return (
        f"No matches with these filters: {used}. Before you tell the user that nothing exists, "
        "you MUST call this tool once more without every filter you guessed (keep only IDs you saw "
        "in a tool result), then check the returned items yourself."
    )


def _validation_message(exc: ValidationError) -> str:
    """Summarize up to 5 Pydantic errors as ``field: message`` for the model to fix."""
    parts = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "arguments"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


# --------------------------------------------------------------------------- result builders
def _refuse(call: ToolCall, code: str, message: str) -> tuple[ToolResult, None]:
    """Pre-check refusal: an error for the model and no log entry (the tool never ran)."""
    logger.info("tool %s refused: %s: %s", call.name, code, message)
    return ToolResult(call_id=call.id, name=call.name, ok=False, error_code=code, error_message=message), None


def _failure(
    call: ToolCall, arguments: dict[str, Any], code: str, message: str
) -> tuple[ToolResult, ToolCallLog]:
    """Invocation failure: an error for the model and a log entry with ``result=None``."""
    logger.info("tool %s failed: %s: %s", call.name, code, message)
    result = ToolResult(call_id=call.id, name=call.name, ok=False, error_code=code, error_message=message)
    log = ToolCallLog(name=call.name, arguments=arguments, result=None, error=f"{code}: {message}")
    return result, log
