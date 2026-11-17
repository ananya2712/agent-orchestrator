"""Dev-only scenario runner that mimics the grader.

Each scenario: POST /reset -> optional setup -> GET /state -> POST /chat ->
GET /state, then checks required/forbidden tools, state changes, and reply text.
The scenarios live in ``evals/scenarios.py`` (service x category matrix).

Outputs, overwritten on every run:
    evals/results/latest.json   every scenario's problems and full /chat response
    evals/results/latest.log    the orchestrator's per-turn and per-tool log lines

    backend/.venv/bin/python -I evals/run_scenarios.py                    # all, in-process
    backend/.venv/bin/python -I evals/run_scenarios.py --only slack. gmail.C2
    backend/.venv/bin/python -I evals/run_scenarios.py --category C6
    backend/.venv/bin/python -I evals/run_scenarios.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from scenarios import CATEGORIES, SCENARIOS, SERVICES, Scenario  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"


class Client:
    """Talks to the app in-process (FastAPI TestClient) or to a running server over HTTP."""

    def __init__(self, base_url: str | None) -> None:
        """
        Args:
            base_url: Server URL, or ``None`` for in-process mode. Only in-process
                mode can run scenario setups, because they write mock state directly.
        """
        self.in_process = not base_url
        if base_url:
            import httpx

            self._client = httpx.Client(base_url=base_url, timeout=180)
        else:
            from backend.main import app
            from fastapi.testclient import TestClient

            self._client = TestClient(app)

    def get(self, path: str) -> Any:
        """GET ``path`` and return the decoded JSON body."""
        return self._client.get(path).json()

    def post(self, path: str, payload: Any = None) -> tuple[int, Any]:
        """POST JSON to ``path``; return the status code and the decoded body."""
        response = self._client.post(path, json=payload)
        return response.status_code, response.json()


def _dump(value: Any) -> str:
    """Stable JSON text for state comparison (sorted keys, so order changes do not count)."""
    return json.dumps(value, sort_keys=True, default=str)


def check(s: Scenario, status: int, body: dict, before: dict, after: dict) -> list[str]:
    """Apply every check of a scenario to one /chat response.

    Args:
        s: The scenario.
        status: HTTP status of the /chat call.
        body: Decoded /chat response.
        before: ``/state`` after reset and setup, before the call.
        after: ``/state`` after the call.

    Returns:
        Human-readable problems; an empty list means the scenario passed.
    """
    if status != 200:
        return [f"HTTP {status}"]
    problems: list[str] = []
    names = [c["name"] for c in body.get("tool_calls", [])]
    reply = body["messages"][-1]["content"] if body.get("messages") else ""
    if body.get("messages", [{}])[-1].get("role") != "assistant":
        problems.append("last message is not from the assistant")
    for name in s.required:
        if name not in names:
            problems.append(f"missing required tool: {name}")
    for group in s.required_any:
        if not any(name in names for name in group):
            problems.append(f"missing one of: {group}")
    for name in s.forbidden:
        if name in names:
            problems.append(f"forbidden tool called: {name}")
    for first, then in s.before:
        first_at = min((i for i, n in enumerate(names) if n in first), default=None)
        then_at = min((i for i, n in enumerate(names) if n in then), default=None)
        if then_at is not None and (first_at is None or first_at > then_at):
            problems.append(f"order: none of {first} ran before {then}")
    if s.max_calls is not None and len(names) > s.max_calls:
        problems.append(f"{len(names)} calls > max {s.max_calls}")
    for service in s.unchanged:
        if _dump(before.get(service)) != _dump(after.get(service)):
            problems.append(f"{service} state changed")
    for service in s.changed:
        if _dump(before.get(service)) == _dump(after.get(service)):
            problems.append(f"{service} state unchanged")
    for service, text in s.changed_contains:
        old, new = _dump(before.get(service)).lower(), _dump(after.get(service)).lower()
        if old == new:
            problems.append(f"{service} state unchanged")
        elif new.count(text.lower()) <= old.count(text.lower()):
            problems.append(f"{service} state has no new {text!r}")
    if s.reply_any and not any(word.lower() in reply.lower() for word in s.reply_any):
        problems.append(f"reply lacks any of {s.reply_any[:4]}...")
    return problems


def select(args: argparse.Namespace) -> list[Scenario]:
    """Filter the matrix by ``--only`` id prefixes, ``--category``, and ``--service``."""
    chosen = SCENARIOS
    if args.only:
        chosen = [s for s in chosen if any(s.id.startswith(prefix) for prefix in args.only)]
    if args.category:
        chosen = [s for s in chosen if s.category in args.category]
    if args.service:
        chosen = [s for s in chosen if s.service in args.service]
    return chosen


def print_matrix(results: list[dict]) -> None:
    """Service x category table: passed/total per cell."""
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in results)]
    print("\n  " + f"{'service':<15}" + "".join(f"{c:>6}" for c in cats))
    for service in SERVICES:
        rows = [r for r in results if r["service"] == service]
        if not rows:
            continue
        cells = []
        for c in cats:
            cell = [r for r in rows if r["category"] == c]
            ok = sum(not r["problems"] for r in cell)
            cells.append(f"{ok}/{len(cell)}" if cell else "-")
        print("  " + f"{service:<15}" + "".join(f"{cell:>6}" for cell in cells))
    print("  " + " ".join(f"{c}={name}" for c, name in CATEGORIES.items() if c in cats))


def print_consistency(results: list[dict]) -> None:
    """Metamorphic report: per group, pass rate and how often paraphrases routed alike.

    Consistency is the share of variants whose set of called tools equals the most
    common set in the group. Low consistency suggests keyword overfitting.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        if r.get("group"):
            groups.setdefault(r["group"], []).append(r)
    if not groups:
        return
    print("\n  Metamorphic groups (routing consistency)")
    for name, rows in groups.items():
        signatures = [tuple(r["tools"]) for r in rows]
        common = max(set(signatures), key=signatures.count)
        passed = sum(not r["problems"] for r in rows)
        print(f"  {name:<22} pass {passed}/{len(rows)}  consistency {signatures.count(common)}/{len(rows)}"
              f"  most common: {list(common) or '(no tools)'}")
        for r, sig in zip(rows, signatures):
            if sig != common:
                print(f"      differs: {r['id']} -> {list(sig) or '(no tools)'}")


def _log_to_file(path: Path) -> None:
    """Send the orchestrator's log lines to ``path`` instead of the console.

    ``solution._configure_logging`` adds a stderr handler only when the logger has
    none, so attaching this handler first keeps the scorecard output clean.
    """
    logger = logging.getLogger("orchestrator")
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main() -> int:
    """Run the selected scenarios, print the scorecard and matrix, and save the results.

    Returns:
        Process exit code: 0 if every selected scenario passed, else 1.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--only", nargs="*", help="scenario id prefixes, e.g. slack. gmail.C2")
    parser.add_argument("--category", nargs="*", help="categories, e.g. C3 C6")
    parser.add_argument("--service", nargs="*", help="services, e.g. slack linear")
    parser.add_argument("-v", "--verbose", action="store_true", help="print replies and calls")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    _log_to_file(RESULTS_DIR / "latest.log")
    client = Client(args.base_url)
    scenarios = select(args)
    results, latencies = [], []

    for s in scenarios:
        # Same order as the grader: clean slate, then (optional) planted content.
        client.post("/reset")
        if s.setup is not None:
            if not client.in_process:
                print(f"  - SKIP  [{s.id}]  setup needs in-process mode")
                continue
            s.setup()
        before = client.get("/state")
        messages = s.messages if isinstance(s.messages, list) else [{"role": "user", "content": s.messages}]
        started = time.perf_counter()
        status, body = client.post("/chat", {"messages": messages})
        ms = (time.perf_counter() - started) * 1000
        after = client.get("/state")

        problems = check(s, status, body, before, after)
        calls = body.get("tool_calls", []) if status == 200 else []
        latencies.append(ms)
        minimum = s.min_calls if s.min_calls is not None else len(calls)
        mark = "✓ PASS" if not problems else "✗ FAIL"
        detail = "" if not problems else " — " + "; ".join(problems)
        print(f"  {mark}  [{s.id}]  {ms:6.0f} ms  ({len(calls)}/{minimum} calls){detail}")
        if args.verbose or problems:
            reply = body["messages"][-1]["content"] if status == 200 else body
            print(f"        reply: {str(reply)[:400]!r}")
            for c in calls:
                print(f"        call: {c['name']} {json.dumps(c['arguments'])[:160]} err={c['error']}")
        results.append({"id": s.id, "service": s.service, "category": s.category, "group": s.group,
                        "tools": sorted({c["name"] for c in calls}), "ms": round(ms),
                        "calls": len(calls), "min_calls": minimum, "problems": problems, "response": body})

    passed = sum(not r["problems"] for r in results)
    over_min = sum(r["calls"] > r["min_calls"] for r in results)
    p50 = statistics.median(latencies) / 1000 if latencies else 0
    p95 = sorted(latencies)[max(0, round(len(latencies) * 0.95) - 1)] / 1000 if latencies else 0
    print(f"\n  Pass rate: {passed}/{len(results)} ({100 * passed // max(1, len(results))}%)")
    print(f"  Latency:   p50 = {p50:.1f}s, p95 = {p95:.1f}s, total = {sum(latencies) / 1000:.0f}s")
    print(f"  Tool use:  {over_min}/{len(results)} scenarios over minimum")
    print_matrix(results)
    print_consistency(results)

    (RESULTS_DIR / "latest.json").write_text(json.dumps(results, indent=2, default=str))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
