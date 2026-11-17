"""Read-only access to the tool registry, plus the compact index the router reads.

``backend.main`` imports ``backend.solution`` before it defines ``get_tool_spec``,
so this module imports ``backend.main`` lazily, inside functions only.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from functools import lru_cache
from typing import Any

from backend.tooling import ToolSpec

SERVICE_LABELS = {
    "gmail": "Gmail",
    "googlecalendar": "Google Calendar",
    "googledrive": "Google Drive",
    "slack": "Slack",
    "linear": "Linear",
    "github": "GitHub",
    "perplexity": "Perplexity (web search / research)",
}


def _main() -> Any:
    """Import ``backend.main`` at call time (see the module docstring)."""
    from backend import main

    return main


@lru_cache(maxsize=1)
def tool_specs() -> dict[str, ToolSpec]:
    """All registered tools by name. Built once; the registry never changes at runtime."""
    return _main().get_all_tool_specs()


def has_tool(name: str) -> bool:
    """Return True if ``name`` is a registered tool."""
    return name in tool_specs()


@lru_cache(maxsize=1)
def _names_by_lowercase() -> dict[str, str]:
    """Lowercased name -> registered name, for case-insensitive lookups."""
    return {name.lower(): name for name in tool_specs()}


def canonical_name(name: str) -> str | None:
    """Return the registered spelling of ``name``, ignoring case; ``None`` if unknown.

    Example: ``googlecalendar_create_event`` -> ``GOOGLECALENDAR_CREATE_EVENT``.
    """
    return _names_by_lowercase().get(name.strip().lower())


def get_spec(name: str) -> ToolSpec:
    """Return the spec for a registered tool. Raises ``KeyError`` for unknown names."""
    return tool_specs()[name]


@lru_cache(maxsize=1)
def compact_index() -> str:
    """Render every tool as ``- NAME: description``, grouped under service headings.

    This is all the router reads about the tools: about 3.4K tokens for 191 tools,
    against about 72K tokens of full schemas. It is static, so OpenAI prompt
    caching can reuse it across requests.

    Returns:
        The index as Markdown-like text, services in alphabetical order.
    """
    by_service: dict[str, list[ToolSpec]] = defaultdict(list)
    for spec in tool_specs().values():
        by_service[spec.service].append(spec)
    sections = []
    for service in sorted(by_service):
        specs = sorted(by_service[service], key=lambda s: s.name)
        label = SERVICE_LABELS.get(service, service)
        lines = [f"## {label} ({len(specs)} tools)"]
        lines += [f"- {s.name}: {s.description}" for s in specs]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


_ANNOTATION_KEYS = ("title", "description", "default")


@lru_cache(maxsize=None)
def _schema(name: str) -> dict[str, Any]:
    """Strict OpenAI schema for one tool, with every optional field made nullable.

    Strict mode lists every field as required. For an optional field with a
    non-null default (for example ``types: str = "public_channel,...,im"``), the
    model then has to invent a value, and live runs showed it narrowing such
    defaults. Allowing ``null`` lets it say "use the default";
    ``executor.clean_arguments`` drops the null so the tool's own default applies.
    Built once per name, then cached.
    """
    schema = copy.deepcopy(_main().get_openai_tools([name])[0])
    fields = get_spec(name).args_model.model_fields
    properties = schema["parameters"].get("properties", {})
    for field_name, prop in properties.items():
        field = fields.get(field_name)
        if field is None or field.is_required() or _allows_null(prop):
            continue
        # Move the type keywords into the first anyOf branch; keep annotations outside.
        type_part = {k: v for k, v in prop.items() if k not in _ANNOTATION_KEYS}
        annotations = {k: v for k, v in prop.items() if k in _ANNOTATION_KEYS}
        annotations["description"] = (
            f"{annotations.get('description', '')} Send null to use the default, "
            "unless the user asked for a specific value."
        ).strip()
        properties[field_name] = {**annotations, "anyOf": [type_part, {"type": "null"}]}
    return schema


def _allows_null(prop: dict[str, Any]) -> bool:
    """True if a property schema already accepts null."""
    branches = prop.get("anyOf", [])
    return prop.get("type") == "null" or any(b.get("type") == "null" for b in branches)


def openai_schemas(names: list[str]) -> list[dict[str, Any]]:
    """Return the tool schemas the agent sees for the given names.

    Args:
        names: Tool names, in the order the router chose them. Unknown names
            are skipped.

    Returns:
        Responses-API function schemas, in the same order.
    """
    # Never call get_openai_tools([]): an empty list returns all 191 tools.
    return [_schema(name) for name in names if has_tool(name)]
