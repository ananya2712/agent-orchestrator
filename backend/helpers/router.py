"""Tool routing: one structured LLM call over a compact index of all tools.

The router never sees full schemas. It reads one line per tool (about 3.4K
tokens for 191 tools, versus about 72K tokens of full schemas) and returns the
names the agent may need. Code validates the names and caps the list.

The router instructions are identical on every call (the per-call limit goes in
the user message), so OpenAI prompt caching reuses the whole catalog prefix.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from backend.helpers import catalog
from backend.helpers.config import Settings
from backend.helpers.contracts import ModelAdapter, ModelError
from backend.helpers.prompts import router_instructions

logger = logging.getLogger("orchestrator")

ROUTER_MAX_OUTPUT_TOKENS = 400  # 12 tool names need well under 200
TRANSCRIPT_CHARS_PER_MESSAGE = 1_500
EXPAND_LIMIT = 6

# Lookup tools a write cannot work without, and sibling searches over the same
# kind of item in another service. When the router picks the key but not its
# companion, code adds the companion (live runs showed the router missing them).
# Keys and values are catalog tool names only, never fixture data.
COMPANION_TOOLS: dict[str, tuple[str, ...]] = {
    "slack_send_message": ("slack_list_conversations",),
    "linear_create_issue": ("linear_list_teams", "linear_list_projects"),
    "linear_update_issue": ("linear_list_issues", "linear_list_issue_statuses"),
    "linear_create_comment": ("linear_list_issues",),
    "GMAIL_REPLY_TO_THREAD": ("GMAIL_FETCH_EMAILS",),
    "GOOGLEDRIVE_EDIT_FILE": ("GOOGLEDRIVE_FIND_FILE",),
    "GOOGLEDRIVE_COPY_FILE": ("GOOGLEDRIVE_FIND_FILE",),
    "GOOGLEDRIVE_CREATE_COMMENT": ("GOOGLEDRIVE_FIND_FILE",),
    "GOOGLECALENDAR_PATCH_EVENT": ("GOOGLECALENDAR_FIND_EVENT",),
    "GOOGLECALENDAR_DELETE_EVENT": ("GOOGLECALENDAR_FIND_EVENT",),
    # GitHub tools need owner + repo: add the repository search (and, for PRs, the
    # cross-repository search) so a partial or missing repo name can be resolved.
    "github_list_pull_requests": ("github_search_repositories", "github_search_pull_requests"),
    "github_list_issues": ("github_search_repositories",),
    "github_issue_read": ("github_search_repositories",),
    "github_issue_write": ("github_search_repositories",),
    "github_add_issue_comment": ("github_search_repositories",),
    "github_list_commits": ("github_search_repositories",),
    "github_get_file_contents": ("github_search_repositories",),
    # Documents live in Drive and in Linear; search both.
    "linear_list_documents": ("GOOGLEDRIVE_FIND_FILE",),
    "GOOGLEDRIVE_FIND_FILE": ("linear_list_documents",),
}
MAX_COMPANIONS = 4

# Overlapping capabilities: one kind of item lives in several services. When the
# user names no service, every service that holds the item must be searchable,
# so code adds each one's main search tool. Live runs showed the router reading
# "discussions" as GitHub Discussions only. Catalog tool names only.
ITEM_SEARCH_TOOLS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("discussion", "conversation", "message", "thread", "chat"): ("slack_search_messages", "GMAIL_FETCH_EMAILS"),
    ("ticket", "issue", "bug"): ("linear_list_issues", "github_search_issues"),
    ("document", "doc ", "docs"): ("GOOGLEDRIVE_FIND_FILE", "linear_list_documents"),
}
# Any of these words means the user named a service, so no expansion happens.
SERVICE_WORDS = (
    "slack", "channel", " dm", "gmail", "email", "e-mail", "inbox", "mail", "draft", "calendar",
    "meeting", "event", "drive", "linear", "github", "repo", "pull request", "perplexity", "web",
)
MAX_ITEM_SEARCHES = 4

# Used only when the router call fails for a non-fatal reason.
_FALLBACK_KEYWORDS = {
    "gmail": ("email", "e-mail", "gmail", "inbox", "mail", "draft"),
    "googlecalendar": ("calendar", "meeting", "schedule", "event", "invite", "availability"),
    "googledrive": ("drive", "file", "folder", "document", "spreadsheet"),
    "slack": ("slack", "channel", "dm"),
    "linear": ("linear", "issue", "ticket", "project", "cycle"),
    "github": ("github", "repo", "pull request", "commit", "branch"),
    "perplexity": ("search the web", "research", "perplexity", "news"),
}
_FALLBACK_GITHUB = (
    "github_get_me",
    "github_search_repositories",
    "github_list_issues",
    "github_issue_read",
    "github_issue_write",
    "github_list_pull_requests",
    "github_pull_request_read",
    "github_search_issues",
    "github_list_commits",
    "github_get_file_contents",
)
_FALLBACK_MAX_TOOLS = 40


@dataclass(frozen=True)
class RouteResult:
    """The router's answer for one turn.

    Attributes:
        tools: Valid tool names for the agent, at most ``max_routed_tools``.
        source: ``"router"`` (LLM choice) or ``"fallback"`` (keyword match).
        dropped: Names the router returned that do not exist (for the log).
        latency_ms: Wall-clock time of the routing step.
        usage: Token counts of the router call (empty for the fallback).
        search_all: Search tools the agent must call before answering, because the
            request names no service and its kind of item lives in several
            services. Empty for most requests.
    """

    tools: list[str]
    source: str  # "router" | "fallback"
    dropped: list[str]
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    search_all: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def router_schema() -> dict[str, Any]:
    """Strict output schema whose items are an ``enum`` of every registered tool name.

    Structured outputs then cannot produce an invented, misspelled, or wrongly
    cased name, or a service name instead of a tool name (all seen in live runs).
    """
    return {
        "type": "object",
        "properties": {"tools": {"type": "array", "items": {"type": "string", "enum": sorted(catalog.tool_specs())}}},
        "required": ["tools"],
        "additionalProperties": False,
    }


@lru_cache(maxsize=1)
def _instructions() -> str:
    """The router system prompt, built once so it is byte-identical on every call."""
    return router_instructions(catalog.compact_index())


def with_companions(selected: list[str]) -> list[str]:
    """Append the lookup tools that the selected writes depend on (see ``COMPANION_TOOLS``).

    Args:
        selected: Router-chosen tool names.

    Returns:
        ``selected`` plus up to ``MAX_COMPANIONS`` missing lookups, in a stable order.
    """
    extra: list[str] = []
    for name in selected:
        for companion in COMPANION_TOOLS.get(name, ()):
            if companion not in selected and companion not in extra and catalog.has_tool(companion):
                extra.append(companion)
    return selected + extra[:MAX_COMPANIONS]


def item_searches(user_text: str) -> list[str]:
    """Every service's search tool for the kinds of items an unscoped request mentions.

    Args:
        user_text: The latest user message.

    Returns:
        An empty list when the user named a service (``SERVICE_WORDS``) or named
        no known kind of item; otherwise up to ``MAX_ITEM_SEARCHES`` search tools
        from ``ITEM_SEARCH_TOOLS``, in a stable order.
    """
    text = f" {user_text.lower()} "
    if any(word in text for word in SERVICE_WORDS):
        return []
    searches: list[str] = []
    for words, tools in ITEM_SEARCH_TOOLS.items():
        if any(word in text for word in words):
            searches += [t for t in tools if t not in searches and catalog.has_tool(t)]
    return searches[:MAX_ITEM_SEARCHES]


def with_item_searches(selected: list[str], user_text: str) -> list[str]:
    """Return ``selected`` plus the missing tools from ``item_searches(user_text)``."""
    return selected + [t for t in item_searches(user_text) if t not in selected]


def transcript(items: list[dict[str, Any]]) -> str:
    """Render messages as ``ROLE: text`` blocks, each cut to a fixed length.

    Args:
        items: Role messages only (no tool items), oldest first.

    Returns:
        Plain text for the router's user message.
    """
    lines = []
    for item in items:
        content = item["content"]
        if len(content) > TRANSCRIPT_CHARS_PER_MESSAGE:
            content = content[:TRANSCRIPT_CHARS_PER_MESSAGE] + " ...[truncated]"
        lines.append(f"{item['role'].upper()}: {content}")
    return "\n\n".join(lines)


class ToolRouter:
    """Picks the small tool subset that the agent sees for one turn."""

    def __init__(self, model: ModelAdapter, settings: Settings) -> None:
        """
        Args:
            model: Adapter for the router model (a structured-output call).
            settings: Supplies the tool cap and the history window.
        """
        self.model = model
        self.settings = settings

    def route(self, history: list[dict[str, Any]]) -> RouteResult:
        """Select tools for the latest user message.

        Args:
            history: The whole conversation. Only the last
                ``router_history_messages`` role messages are sent, so follow-ups
                ("the second one") route with their context.

        Returns:
            The selected tools. A non-fatal failure returns the keyword fallback.

        Raises:
            ModelError: Only when ``fatal`` is set (bad key, no quota). Retrying
                cannot help, so the caller should reply at once.
        """
        recent = [item for item in history if "role" in item][-self.settings.router_history_messages :]
        limit = self.settings.max_routed_tools
        prompt = (
            f"Conversation (oldest first):\n\n{transcript(recent)}\n\n"
            f"Select at most {limit} tools to complete the user's current request."
        )
        started = time.perf_counter()
        try:
            selected, dropped = self._select(prompt, limit, exclude=())
        except ModelError as exc:
            if exc.fatal:
                raise
            return self._fallback(recent, exc)
        except Exception as exc:  # malformed JSON or an unexpected bug
            return self._fallback(recent, exc)
        latest_user = next((i["content"] for i in reversed(recent) if i["role"] == "user"), "")
        return RouteResult(
            tools=with_item_searches(with_companions(selected), latest_user),
            search_all=item_searches(latest_user),
            source="router",
            dropped=dropped,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=dict(getattr(self.model, "last_usage", {}) or {}),
        )

    def expand(self, need: str, current: list[str]) -> list[str]:
        """Tool loader for the agent's ``request_tools`` meta-tool. Never raises.

        Args:
            need: The missing capability, in the agent's words.
            current: Tools the agent already has; they are never returned again.

        Returns:
            Up to ``EXPAND_LIMIT`` new tool names, or an empty list on failure.
        """
        prompt = (
            f"The assistant already has these tools: {', '.join(current) or '(none)'}.\n"
            f"It needs a tool for: {need}\n"
            f"Select at most {EXPAND_LIMIT} additional tools that cover this need."
        )
        try:
            added, _ = self._select(prompt, EXPAND_LIMIT, exclude=current)
        except Exception as exc:
            logger.warning("request_tools routing failed: %s", exc)
            return []
        return added

    def _select(
        self, prompt: str, limit: int, exclude: tuple[str, ...] | list[str]
    ) -> tuple[list[str], list[str]]:
        """Run one router call and clean its output.

        Args:
            prompt: User message for the router.
            limit: Most names to keep.
            exclude: Names to skip (already active).

        Returns:
            ``(selected, dropped)``: valid unique names in model order, capped at
            ``limit``; and the names that are not in the registry.

        Raises:
            ModelError: The provider call failed.
        """
        data = self.model.complete_json(
            instructions=_instructions(),
            items=[{"role": "user", "content": prompt}],
            schema=router_schema(),
            schema_name="tool_selection",
            max_output_tokens=ROUTER_MAX_OUTPUT_TOKENS,
        )
        selected: list[str] = []
        dropped: list[str] = []
        # Keep model order (it reflects priority); drop invented names and duplicates.
        # The enum should make every name valid; canonical_name() is a second guard
        # (for example, if a provider ignores the enum and returns lowercase).
        for raw in data.get("tools", []):
            name = catalog.canonical_name(raw) if isinstance(raw, str) else None
            if name is None:
                dropped.append(str(raw))
            elif name not in selected and name not in exclude:
                selected.append(name)
        return selected[:limit], dropped

    def _fallback(self, recent: list[dict[str, Any]], exc: Exception) -> RouteResult:
        """Pick tools by keyword when the router call fails for a transient reason.

        A service matches when any of its keywords appears in the user's recent
        messages. A matched service adds all its tools, except GitHub, which adds
        a fixed 10-tool subset because it has 86 tools. The total is capped.
        """
        logger.warning("router failed, using keyword fallback: %s", exc)
        user_text = " ".join(i["content"] for i in recent if i["role"] == "user").lower()
        names: list[str] = []
        for service, keywords in _FALLBACK_KEYWORDS.items():
            if not any(keyword in user_text for keyword in keywords):
                continue
            if service == "github":
                names += [n for n in _FALLBACK_GITHUB if catalog.has_tool(n)]
            else:
                names += sorted(n for n, s in catalog.tool_specs().items() if s.service == service)
        return RouteResult(tools=names[:_FALLBACK_MAX_TOOLS], source="fallback", dropped=[])
