"""Dry run of the /chat orchestrator with a scripted model. No API key needed.

The tools are the real in-process mocks, so state changes are real too.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from backend.chat_schema import ChatRequest
from backend.helpers.config import Settings
from backend.helpers.contracts import ModelError, ModelTurn, ToolCall
from backend.helpers.model_adapter import OpenAIResponsesAdapter, to_responses_input
from backend.helpers.orchestrator import REQUEST_TOOLS, Orchestrator, history_from_messages
from backend.helpers.router import ToolRouter
from backend.main import app, get_tool_spec, reset_all_mock_state
from fastapi.testclient import TestClient
from openai.types.responses import ResponseFunctionToolCall

SETTINGS = Settings(agent_model="test", router_model="test", max_steps=4, max_tool_calls=3)


class ScriptedModel:
    """Returns scripted turns in order and records what it was sent."""

    def __init__(self, turns=(), routes=()) -> None:
        self.turns = list(turns)
        self.routes = list(routes)
        self.calls: list[dict] = []

    def complete(self, *, instructions, items, tools, tool_choice="auto"):
        self.calls.append(
            {"items": [dict(i) for i in items], "tools": [t["name"] for t in tools], "tool_choice": tool_choice}
        )
        turn = self.turns.pop(0) if self.turns else ModelTurn(content="(script ended)")
        if isinstance(turn, Exception):
            raise turn
        return turn

    def complete_json(self, *, instructions, items, schema, schema_name, max_output_tokens=None):
        self.calls.append({"router_prompt": items[0]["content"]})
        route = self.routes.pop(0)
        if isinstance(route, Exception):
            raise route
        return {"tools": route}


_ids = iter(range(10_000))


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)


def tools(*calls: ToolCall) -> ModelTurn:
    return ModelTurn(tool_calls=calls)


def text(content: str) -> ModelTurn:
    return ModelTurn(content=content)


def user(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def tool_results(model_call: dict) -> list[dict]:
    return [i for i in model_call["items"] if i.get("type") == "tool_result"]


class OrchestratorDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_all_mock_state()

    def run_loop(self, model: ScriptedModel, tool_names: list[str], loader=None, settings=SETTINGS):
        orchestrator = Orchestrator(model, settings, tool_loader=loader)
        return orchestrator.run(instructions="test", history=user("hi"), tool_names=tool_names)

    def test_direct_answer_runs_no_tools(self) -> None:
        model = ScriptedModel([text("Hello!")])
        result = self.run_loop(model, [])
        self.assertEqual(("Hello!", [], "complete", 1), (result.answer, result.tool_calls, result.stop_reason, result.steps))

    def test_one_tool_call_is_logged_and_its_result_returns_with_the_same_call_id(self) -> None:
        first = call("slack_list_conversations")
        model = ScriptedModel([tools(first), text("You have channels.")])
        result = self.run_loop(model, ["slack_list_conversations"])
        self.assertEqual(["slack_list_conversations"], [log.name for log in result.tool_calls])
        self.assertIn("conversations", result.tool_calls[0].result)
        self.assertIsNone(result.tool_calls[0].error)
        [observation] = tool_results(model.calls[1])
        self.assertEqual(first.id, observation["call_id"])
        self.assertEqual(["slack_list_conversations"], model.calls[0]["tools"])

    def test_two_calls_in_one_turn_run_once_each_in_request_order(self) -> None:
        a, b = call("slack_list_users"), call("slack_list_conversations")
        model = ScriptedModel([tools(a, b), text("done")])
        result = self.run_loop(model, ["slack_list_users", "slack_list_conversations"])
        self.assertEqual(["slack_list_users", "slack_list_conversations"], [log.name for log in result.tool_calls])
        kinds = [i.get("type") for i in model.calls[1]["items"][1:]]
        self.assertEqual(["tool_call", "tool_call", "tool_result", "tool_result"], kinds)

    def test_dependent_calls_change_state(self) -> None:
        channels = get_tool_spec("slack_list_conversations").invoke().model_dump(mode="json")["result"]
        channel_id = next(c["id"] for c in channels["conversations"] if c["name"] == "launch-war-room")
        model = ScriptedModel(
            [
                tools(call("slack_list_conversations")),
                tools(call("slack_send_message", text="Launch summary", channel=channel_id, thread_ts=None)),
                text("Posted."),
            ]
        )
        result = self.run_loop(model, ["slack_list_conversations", "slack_send_message"])
        self.assertEqual("complete", result.stop_reason)
        self.assertEqual({"text": "Launch summary", "channel": channel_id}, result.tool_calls[1].arguments)
        history = get_tool_spec("slack_conversations_history").invoke(channel=channel_id).model_dump(mode="json")
        self.assertIn("Launch summary", json.dumps(history))

    def test_unknown_tool_does_not_run_and_is_not_logged(self) -> None:
        model = ScriptedModel([tools(call("delete_everything")), text("Sorry.")])
        result = self.run_loop(model, [])
        self.assertEqual([], result.tool_calls)
        self.assertIn("unknown_tool", tool_results(model.calls[1])[0]["output"])

    def test_invalid_arguments_are_logged_with_an_error_and_the_model_can_recover(self) -> None:
        model = ScriptedModel(
            [tools(call("slack_list_conversations", limit="many")), tools(call("slack_list_conversations")), text("ok")]
        )
        result = self.run_loop(model, ["slack_list_conversations"])
        self.assertTrue(result.tool_calls[0].error.startswith("invalid_arguments"))
        self.assertIsNone(result.tool_calls[0].result)
        self.assertIsNone(result.tool_calls[1].error)

    def test_tool_exception_becomes_tool_failed_without_crashing(self) -> None:
        model = ScriptedModel([tools(call("GOOGLEDRIVE_DOWNLOAD_FILE", file_id="budget_2025.xlsx")), text("Not found.")])
        result = self.run_loop(model, ["GOOGLEDRIVE_DOWNLOAD_FILE"])
        self.assertTrue(result.tool_calls[0].error.startswith("tool_failed"))
        self.assertIn("tool_failed", tool_results(model.calls[1])[0]["output"])
        self.assertEqual("complete", result.stop_reason)

    def test_repeated_identical_call_runs_once(self) -> None:
        model = ScriptedModel(
            [tools(call("slack_list_users")), tools(call("slack_list_users")), text("done")]
        )
        result = self.run_loop(model, ["slack_list_users"])
        self.assertEqual(1, len(result.tool_calls))
        self.assertIn("repeated_tool_call", tool_results(model.calls[2])[-1]["output"])

    def test_step_limit_stops_and_forces_a_final_answer_without_tools(self) -> None:
        turns = [tools(call("slack_list_conversations", limit=n)) for n in range(1, 10)]
        model = ScriptedModel(turns)
        model.turns.insert(SETTINGS.max_steps, text("Partial answer."))
        settings = replace(SETTINGS, max_tool_calls=50)  # isolate the step limit
        result = self.run_loop(model, ["slack_list_conversations"], settings=settings)
        self.assertEqual("step_limit", result.stop_reason)
        self.assertEqual(SETTINGS.max_steps, result.steps)
        self.assertEqual("none", model.calls[-1]["tool_choice"])
        self.assertEqual("Partial answer.", result.answer)

    def test_tool_call_limit_refuses_extra_calls_in_the_same_turn(self) -> None:
        batch = tools(*(call("slack_list_conversations", limit=n) for n in range(1, 6)))
        model = ScriptedModel([batch, text("Stopped early.")])
        result = self.run_loop(model, ["slack_list_conversations"])
        self.assertEqual(SETTINGS.max_tool_calls, len(result.tool_calls))
        self.assertEqual("tool_call_limit", result.stop_reason)
        self.assertEqual(5, len(tool_results(model.calls[-1])))  # every call got an output

    def test_empty_model_turn_is_invalid_and_still_gets_an_answer(self) -> None:
        model = ScriptedModel([ModelTurn(), text("Recovered.")])
        result = self.run_loop(model, [])
        self.assertEqual(("invalid_model_response", "Recovered."), (result.stop_reason, result.answer))

    def test_model_error_keeps_the_calls_already_made(self) -> None:
        model = ScriptedModel([tools(call("slack_list_users")), ModelError("boom", fatal=False)])
        result = self.run_loop(model, ["slack_list_users"])
        self.assertEqual("model_error", result.stop_reason)
        self.assertEqual(["slack_list_users"], [log.name for log in result.tool_calls])
        self.assertIn("slack_list_users", result.answer)

    def test_null_optional_arguments_are_dropped_before_invoke_and_in_the_log(self) -> None:
        model = ScriptedModel(
            [tools(call("GMAIL_FETCH_EMAILS", query="revenue", label_ids=None, page_token=None, max_results=2)), text("ok")]
        )
        result = self.run_loop(model, ["GMAIL_FETCH_EMAILS"])
        self.assertEqual({"query": "revenue", "max_results": 2}, result.tool_calls[0].arguments)
        self.assertIsNone(result.tool_calls[0].error)

    def test_request_tools_adds_schemas_and_is_not_logged(self) -> None:
        loader = mock.Mock(return_value=["linear_list_projects"])
        model = ScriptedModel(
            [tools(call(REQUEST_TOOLS, need="list Linear projects")), tools(call("linear_list_projects")), text("ok")]
        )
        result = self.run_loop(model, [], loader=loader)
        self.assertEqual([REQUEST_TOOLS], model.calls[0]["tools"])
        self.assertIn("linear_list_projects", model.calls[1]["tools"])
        self.assertEqual(["linear_list_projects"], [log.name for log in result.tool_calls])
        loader.assert_called_once_with("list Linear projects", [])


class RouterDryRunTests(unittest.TestCase):
    def test_router_drops_unknown_names_dedupes_and_caps(self) -> None:
        names = ["slack_list_users", "made_up_tool", "slack_list_users"] + [
            # Tools without companions, so only the router cap decides the length.
            f"github_{n}" for n in ("get_me", "list_branches", "list_tags", "list_releases", "get_commit",
                                    "list_label", "get_label", "list_gists", "get_gist", "get_tag",
                                    "list_starred_repositories", "get_latest_release")
        ]
        model = ScriptedModel(routes=[names])
        route = ToolRouter(model, SETTINGS).route(user("x"))
        self.assertEqual(SETTINGS.max_routed_tools, len(route.tools))
        self.assertEqual(1, route.tools.count("slack_list_users"))
        self.assertEqual(["made_up_tool"], route.dropped)

    def test_router_sees_recent_history_for_follow_ups(self) -> None:
        model = ScriptedModel(routes=[["GOOGLECALENDAR_CREATE_EVENT"]])
        history = user("Schedule a meeting") + [{"role": "assistant", "content": "Which project?"}] + user("Tool Orchestrator")
        ToolRouter(model, SETTINGS).route(history)
        prompt = model.calls[0]["router_prompt"]
        self.assertIn("Which project?", prompt)
        self.assertIn("USER: Tool Orchestrator", prompt)

    def test_non_fatal_router_error_falls_back_to_keywords(self) -> None:
        model = ScriptedModel(routes=[ModelError("timeout", fatal=False)])
        route = ToolRouter(model, SETTINGS).route(user("What conversations do I have in Slack?"))
        self.assertEqual("fallback", route.source)
        self.assertIn("slack_list_conversations", route.tools)

    def test_fatal_router_error_is_raised(self) -> None:
        model = ScriptedModel(routes=[ModelError("no quota", fatal=True)])
        with self.assertRaises(ModelError):
            ToolRouter(model, SETTINGS).route(user("hi"))


class CatalogDryRunTests(unittest.TestCase):
    def test_optional_fields_with_defaults_accept_null_and_required_fields_do_not_change(self) -> None:
        from backend.helpers import catalog

        props = catalog.openai_schemas(["slack_list_conversations"])[0]["parameters"]["properties"]
        self.assertEqual([{"type": "string"}, {"type": "null"}], props["types"]["anyOf"])
        draft = catalog.openai_schemas(["GMAIL_CREATE_EMAIL_DRAFT"])[0]["parameters"]["properties"]
        self.assertNotIn("Send null", draft["recipient_email"]["description"])  # required field

    def test_canonical_name_ignores_case(self) -> None:
        from backend.helpers import catalog

        self.assertEqual("GOOGLECALENDAR_CREATE_EVENT", catalog.canonical_name("googlecalendar_create_event"))
        self.assertIsNone(catalog.canonical_name("Linear"))

    def test_unscoped_item_search_adds_every_service_but_a_named_service_does_not(self) -> None:
        from backend.helpers.router import with_item_searches

        broad = with_item_searches(["github_list_discussions"], "Find all discussions and tickets about the launch.")
        for tool in ("slack_search_messages", "GMAIL_FETCH_EMAILS", "linear_list_issues", "github_search_issues"):
            self.assertIn(tool, broad)
        self.assertEqual(["slack_search_messages"], with_item_searches(["slack_search_messages"], "Find Slack messages"))

    def test_router_adds_the_lookup_a_write_needs(self) -> None:
        model = ScriptedModel(routes=[["slack_send_message"]])
        route = ToolRouter(model, SETTINGS).route(user("DM Morgan"))
        self.assertEqual(["slack_send_message", "slack_list_conversations"], route.tools)


class EngineEdgeCaseTests(unittest.TestCase):
    """Orchestrator-level failure modes, one test per case (no API)."""

    def setUp(self) -> None:
        reset_all_mock_state()

    def run_loop(self, model: ScriptedModel, tool_names: list[str], settings=SETTINGS):
        return Orchestrator(model, settings).run(instructions="test", history=user("hi"), tool_names=tool_names)

    def test_malformed_json_arguments_become_a_controlled_error(self) -> None:
        from backend.helpers.executor import execute
        from backend.helpers.model_adapter import _parse_arguments

        arguments = _parse_arguments('{"limit": ')  # truncated JSON from the model
        result, log = execute(ToolCall("c1", "slack_list_users", arguments))
        self.assertEqual("malformed_arguments", result.error_code)
        self.assertIn("valid JSON", result.error_message)
        self.assertIsNone(log)  # the tool never ran

    def test_unexpected_parameter_is_rejected_before_the_tool_runs(self) -> None:
        from backend.helpers.executor import execute
        from backend.tooling import ToolSpec

        with mock.patch.object(ToolSpec, "invoke") as invoke:
            result, _ = execute(call("slack_list_users", limit=5, bogus="x"))
        self.assertEqual("invalid_arguments", result.error_code)
        self.assertIn("bogus", result.error_message)
        invoke.assert_not_called()

    def test_a_tool_that_keeps_failing_is_refused_after_the_cap(self) -> None:
        settings = replace(SETTINGS, max_steps=8, max_tool_calls=50, max_failures_per_tool=3)
        turns = [tools(call("GOOGLEDRIVE_DOWNLOAD_FILE", file_id=f"missing-{n}")) for n in range(5)]
        model = ScriptedModel(turns + [text("Could not download.")])
        result = self.run_loop(model, ["GOOGLEDRIVE_DOWNLOAD_FILE"], settings)
        self.assertEqual(3, len(result.tool_calls))  # the 4th and 5th were refused, not run
        self.assertIn("tool_retry_limit", tool_results(model.calls[4])[-1]["output"])

    def test_empty_listing_hint_names_the_guessed_filters(self) -> None:
        from backend.helpers.executor import execute

        result, log = execute(call("linear_list_issues", label="bug", query="P0"))
        self.assertTrue(result.ok)
        self.assertIn("query='P0'", result.hint)
        self.assertIn("MUST call this tool once more", result.observation(10_000))
        self.assertNotIn("hint", json.dumps(log.result))  # the log keeps the plain result

    def test_unfiltered_empty_listing_says_nothing_was_found(self) -> None:
        from backend.helpers.executor import empty_result_hint
        from backend.main import get_tool_spec

        self.assertIn("report that nothing was found", empty_result_hint(get_tool_spec("linear_list_issues"), {}))

    def test_a_slow_tool_times_out_without_hanging_the_request(self) -> None:
        import time

        from backend.helpers.executor import execute
        from backend.tooling import ToolSpec

        def slow(self, **kwargs):
            time.sleep(1.0)

        started = time.perf_counter()
        with mock.patch.object(ToolSpec, "invoke", slow):
            result, log = execute(call("slack_list_users"), timeout_s=0.1)
        self.assertLess(time.perf_counter() - started, 0.8)
        self.assertEqual("tool_timeout", result.error_code)
        self.assertTrue(log.error.startswith("tool_timeout"))

    def test_a_tool_the_agent_was_not_given_is_refused_with_a_pointer_to_request_tools(self) -> None:
        model = ScriptedModel([tools(call("slack_send_message", text="hi", channel="C001")), text("ok")])
        result = self.run_loop(model, ["slack_list_users"])
        self.assertEqual([], result.tool_calls)
        self.assertIn("tool_not_loaded", tool_results(model.calls[1])[0]["output"])

    def test_two_steps_of_only_refused_calls_stop_as_loop_detected(self) -> None:
        settings = replace(SETTINGS, max_steps=8)
        model = ScriptedModel([tools(call("slack_list_users")) for _ in range(8)])
        result = self.run_loop(model, ["slack_list_users"], settings)
        self.assertEqual("loop_detected", result.stop_reason)
        self.assertEqual(3, result.steps)  # 1 real call, then 2 refused-only steps
        self.assertEqual("none", model.calls[-1]["tool_choice"])

    def test_refused_calls_count_toward_the_per_tool_cap(self) -> None:
        settings = replace(SETTINGS, max_steps=8, max_tool_calls=50, max_failures_per_tool=2)
        # One valid call per step keeps loop detection out of the way.
        turns = [tools(call("slack_list_users", limit=n + 1), call("linear_list_issues", query=f"project:p{n}"))
                 for n in range(4)]
        model = ScriptedModel(turns + [text("done")])
        self.run_loop(model, ["slack_list_users", "linear_list_issues"], settings)
        self.assertIn("invalid_query_syntax", tool_results(model.calls[2])[-1]["output"])  # 2nd refusal
        self.assertIn("tool_retry_limit", tool_results(model.calls[3])[-1]["output"])  # cap of 2 reached

    def test_turn_deadline_stops_without_another_model_call(self) -> None:
        settings = replace(SETTINGS, turn_deadline_s=0.0)
        model = ScriptedModel([text("never sent")])
        result = self.run_loop(model, [], settings)
        self.assertEqual("deadline", result.stop_reason)
        self.assertEqual([], model.calls)

    def test_a_bare_empty_list_counts_as_an_empty_listing(self) -> None:
        from backend.helpers.contracts import is_empty_listing

        self.assertTrue(is_empty_listing([]))
        self.assertTrue(is_empty_listing({"users": [], "count": 0}))
        self.assertFalse(is_empty_listing({"users": [{"id": "u"}]}))

    def test_step_limit_answer_is_asked_to_report_unfinished_work(self) -> None:
        settings = replace(SETTINGS, max_tool_calls=50)
        model = ScriptedModel([tools(call("slack_list_conversations", limit=n)) for n in range(1, 10)])
        self.run_loop(model, ["slack_list_conversations"], settings)
        nudge = model.calls[-1]["items"][-1]["content"]
        self.assertIn("what was not done", nudge)


class QuerySyntaxGuardTests(unittest.TestCase):
    def test_invented_field_value_syntax_is_refused_before_the_tool_runs(self) -> None:
        from backend.helpers.executor import execute

        for query in ("project:proj_orchestrator", "key:team_eng", "bug is:open"):
            result, log = execute(call("linear_list_users", team="team_eng", query=query))
            self.assertEqual("invalid_query_syntax", result.error_code, query)
            self.assertIsNone(log)

    def test_plain_text_times_urls_and_gmail_operators_pass(self) -> None:
        from backend.helpers.executor import invented_query_syntax

        for query in ("Tool Orchestrator", "standup 10:00", "https://example.com/x", "Note: launch"):
            self.assertIsNone(invented_query_syntax("linear", {"query": query}), query)
        self.assertIsNone(invented_query_syntax("gmail", {"query": "from:cfo@corp.com subject:revenue"}))
        self.assertIsNone(invented_query_syntax("linear", {"description": "see project:abc"}))  # not a search field


class MissingRequiredValueTests(unittest.TestCase):
    def test_null_required_field_is_refused_and_not_logged(self) -> None:
        from backend.helpers.executor import execute

        result, log = execute(call("GMAIL_CREATE_EMAIL_DRAFT", recipient_email=None, subject="s", body="b"))
        self.assertEqual("missing_required_value", result.error_code)
        self.assertIn("recipient_email", result.error_message)
        self.assertIsNone(log)


class AdapterDryRunTests(unittest.TestCase):
    def test_conversation_items_map_to_responses_input(self) -> None:
        items = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"type": "tool_call", "call_id": "c1", "name": "slack_list_users", "arguments": {"limit": 5}},
            {"type": "tool_result", "call_id": "c1", "output": "{}"},
        ]
        converted = to_responses_input(items)
        self.assertEqual("developer", converted[0]["role"])
        self.assertEqual({"type": "function_call", "call_id": "c1", "name": "slack_list_users", "arguments": '{"limit": 5}'}, converted[2])
        self.assertEqual({"type": "function_call_output", "call_id": "c1", "output": "{}"}, converted[3])

    def test_complete_parses_function_calls_and_sends_expected_options(self) -> None:
        response = SimpleNamespace(
            output=[ResponseFunctionToolCall(type="function_call", call_id="c9", name="slack_list_users", arguments='{"limit": 2}')],
            output_text="",
            usage=SimpleNamespace(input_tokens=10, output_tokens=3, input_tokens_details=SimpleNamespace(cached_tokens=4)),
        )
        client = SimpleNamespace(responses=SimpleNamespace(create=mock.Mock(return_value=response)))
        turn = OpenAIResponsesAdapter(client, "gpt-4.1-mini").complete(
            instructions="i", items=user("u"), tools=[{"type": "function", "name": "slack_list_users"}]
        )
        self.assertEqual((ToolCall("c9", "slack_list_users", {"limit": 2}),), turn.tool_calls)
        self.assertIsNone(turn.content)
        self.assertEqual({"input_tokens": 10, "output_tokens": 3, "cached_tokens": 4}, turn.usage)
        sent = client.responses.create.call_args.kwargs
        self.assertEqual((False, 0, True), (sent["store"], sent["temperature"], sent["parallel_tool_calls"]))


class ChatEndpointDryRunTests(unittest.TestCase):
    def test_chat_returns_history_plus_answer_and_the_tool_log(self) -> None:
        import backend.solution as solution

        model = ScriptedModel(
            turns=[tools(call("slack_list_conversations")), text("You have 9 conversations.")],
            routes=[["slack_list_conversations"]],
        )
        components = (SETTINGS, ToolRouter(model, SETTINGS), Orchestrator(model, SETTINGS))
        with mock.patch.object(solution, "_components", return_value=components):
            response = TestClient(app).post("/chat", json={"messages": user("What conversations do I have in Slack?")})
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(["user", "assistant"], [m["role"] for m in body["messages"]])
        self.assertEqual("You have 9 conversations.", body["messages"][-1]["content"])
        self.assertEqual(["slack_list_conversations"], [c["name"] for c in body["tool_calls"]])

    def test_an_orchestrator_crash_still_returns_http_200_with_an_honest_answer(self) -> None:
        import backend.solution as solution

        model = ScriptedModel(routes=[[]])
        broken = mock.Mock(spec=Orchestrator)
        broken.run.side_effect = RuntimeError("bug")
        with mock.patch.object(solution, "_components", return_value=(SETTINGS, ToolRouter(model, SETTINGS), broken)):
            response = TestClient(app).post("/chat", json={"messages": user("hi")})
        self.assertEqual(200, response.status_code)
        self.assertIn("internal error", response.json()["messages"][-1]["content"])

    def test_agent_prompt_uses_the_mock_clock(self) -> None:
        from backend.helpers.prompts import agent_instructions
        from backend.solution import _current_time

        self.assertIn("April 08, 2026", agent_instructions(_current_time()))

    def test_wire_tool_messages_become_assistant_context(self) -> None:
        request = ChatRequest(messages=[{"role": "tool", "content": "{}"}, {"role": "user", "content": "x"}])
        history = history_from_messages([m.model_dump() for m in request.messages])
        self.assertEqual("assistant", history[0]["role"])


if __name__ == "__main__":
    unittest.main()
