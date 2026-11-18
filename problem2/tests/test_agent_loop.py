import pytest
from helpers import RecordingUI, make_agent

from mini_agent.agent.cancel import Cancelled
from mini_agent.config import AgentConfig
from mini_agent.history import validate
from mini_agent.messages import Usage
from mini_agent.providers.errors import FatalError, RetryableError
from mini_agent.providers.mock import MockToolCall, MockTurn


def call(name, **args):
    return MockToolCall(name, args)


def tool_turn(*calls):
    return MockTurn(tool_calls=list(calls))


def results_of(agent, message_index):
    return agent.history[message_index].tool_results


# ----------------------------------------------------------- basic loop


def test_text_only_turn(workdir):
    agent, provider, ui = make_agent([MockTurn(text="Hello there, how can I help?")], workdir)
    result = agent.run_turn("hi")
    assert result.status == "completed" and result.text == "Hello there, how can I help?"
    assert [m.role for m in agent.history] == ["user", "assistant"]
    assert ui.text == "Hello there, how can I help?"
    assert ui.usages == [Usage(input_tokens=10, output_tokens=5)]
    assert validate(agent.history) == []


def test_tool_call_result_is_fed_back_to_the_model(workdir):
    (workdir / "a.txt").write_text("hello file\n")
    agent, provider, ui = make_agent([tool_turn(call("read_file", path="a.txt")), MockTurn(text="It says hello.")], workdir)
    result = agent.run_turn("what is in a.txt?")
    assert result.status == "completed" and result.steps == 2
    assert [m.role for m in agent.history] == ["user", "assistant", "user", "assistant"]
    assert "hello file" in results_of(agent, 2)[0].content
    assert provider.calls == 2
    second_request = provider.requests[1].messages
    assert second_request[-1].tool_results[0].tool_use_id == agent.history[1].tool_uses[0].id
    assert result.usage == Usage(input_tokens=20, output_tokens=10)


def test_multiple_tool_calls_in_one_reply_all_get_results_in_order(workdir):
    (workdir / "a.txt").write_text("A\n")
    script = [
        tool_turn(
            call("read_file", path="a.txt"),
            call("read_file", path="missing.txt"),
            call("no_such_tool"),
            call("write_file", path="out.txt", content="x\n"),
        ),
        MockTurn(text="done"),
    ]
    agent, provider, ui = make_agent(script, workdir)
    result = agent.run_turn("do things")
    assert result.status == "completed"
    uses = agent.history[1].tool_uses
    results = results_of(agent, 2)
    assert [r.tool_use_id for r in results] == [u.id for u in uses]
    assert [r.is_error for r in results] == [False, True, True, False]
    assert (workdir / "out.txt").read_text() == "x\n"
    assert "Unknown tool" in results[2].content
    assert validate(agent.history) == []  # and the mock accepted the follow-up request


def test_multi_turn_conversation_keeps_context(workdir):
    agent, provider, ui = make_agent([MockTurn(text="first answer"), MockTurn(text="second answer")], workdir)
    agent.run_turn("question one")
    agent.run_turn("question two")
    sent = provider.requests[1].messages
    assert [m.role for m in sent] == ["user", "assistant", "user"]
    assert sent[0].text == "question one" and sent[1].text == "first answer" and sent[2].text == "question two"
    assert len(ui.usages) == 2  # token usage is shown after every turn


def test_tool_calls_and_results_are_shown_to_the_user(workdir):
    (workdir / "a.txt").write_text("x\n")
    agent, _, ui = make_agent([tool_turn(call("read_file", path="a.txt")), MockTurn(text="ok")], workdir)
    agent.run_turn("go")
    assert ui.tool_calls == [("read_file", {"path": "a.txt"})]
    assert ui.tool_results[0][0] == "read_file" and ui.tool_results[0][2] is False


def test_empty_tool_output_is_replaced_so_the_result_is_never_blank(workdir):
    (workdir / "empty.txt").write_text("")
    agent, _, _ = make_agent([tool_turn(call("grep", pattern="zzz")), MockTurn(text="ok")], workdir)
    agent.run_turn("go")
    assert results_of(agent, 2)[0].content.strip() != ""


# ------------------------------------------------------- confirmation


def test_user_rejection_is_reported_to_the_model_and_the_session_continues(workdir):
    ui = RecordingUI(answers=[False])
    script = [tool_turn(call("write_file", path="new.txt", content="data\n")), MockTurn(text="Understood, I will not write it.")]
    agent, provider, _ = make_agent(script, workdir, ui=ui)
    result = agent.run_turn("create new.txt")
    assert result.status == "completed" and result.text.startswith("Understood")
    assert not (workdir / "new.txt").exists()
    rejected = results_of(agent, 2)[0]
    assert rejected.is_error and "rejected" in rejected.content
    assert ui.confirms[0][0] == "write_file" and "Create new.txt" in ui.confirms[0][1]
    assert provider.calls == 2


def test_read_only_tools_never_ask_for_confirmation(workdir):
    (workdir / "a.txt").write_text("x\n")
    script = [tool_turn(call("read_file", path="a.txt"), call("list_files"), call("grep", pattern="x")), MockTurn(text="ok")]
    agent, _, ui = make_agent(script, workdir)
    agent.run_turn("go")
    assert ui.confirms == []


def test_write_edit_and_shell_all_ask_first(workdir):
    (workdir / "a.txt").write_text("old\n")
    script = [
        tool_turn(
            call("write_file", path="b.txt", content="b\n"),
            call("edit_file", path="a.txt", old_string="old", new_string="new"),
            call("run_shell", command="echo hi"),
        ),
        MockTurn(text="ok"),
    ]
    agent, _, ui = make_agent(script, workdir)
    agent.run_turn("go")
    assert [c[0] for c in ui.confirms] == ["write_file", "edit_file", "run_shell"]
    assert "-old" in ui.confirms[1][1] and "+new" in ui.confirms[1][1]
    assert ui.confirms[2][1] == "$ echo hi"
    assert (workdir / "a.txt").read_text() == "new\n"


def test_rejected_shell_command_is_not_executed(workdir):
    ui = RecordingUI(default=False)
    script = [tool_turn(call("run_shell", command="echo hi > made.txt")), MockTurn(text="ok")]
    agent, _, _ = make_agent(script, workdir, ui=ui)
    agent.run_turn("go")
    assert not (workdir / "made.txt").exists()


def test_auto_yes_skips_confirmation(workdir):
    script = [tool_turn(call("write_file", path="b.txt", content="b\n")), MockTurn(text="ok")]
    agent, _, ui = make_agent(script, workdir, config=AgentConfig(auto_yes=True))
    agent.run_turn("go")
    assert ui.confirms == [] and (workdir / "b.txt").read_text() == "b\n"


def test_an_invalid_edit_fails_before_asking_the_user(workdir):
    (workdir / "a.txt").write_text("x\n")
    script = [tool_turn(call("edit_file", path="a.txt", old_string="nope", new_string="y")), MockTurn(text="ok")]
    agent, _, ui = make_agent(script, workdir)
    agent.run_turn("go")
    assert ui.confirms == []
    assert results_of(agent, 2)[0].is_error and "not found" in results_of(agent, 2)[0].content


# ------------------------------------------------------ stop conditions


def test_step_limit_stops_the_turn_and_leaves_a_valid_history(workdir):
    script = [tool_turn(call("list_files")) for _ in range(3)] + [MockTurn(text="continuing")]
    agent, provider, ui = make_agent(script, workdir, config=AgentConfig(max_steps=3))
    result = agent.run_turn("loop forever")
    assert result.status == "max_steps" and result.steps == 3
    assert provider.calls == 3
    assert agent.history[-1].role == "assistant" and "maximum of 3 steps" in agent.history[-1].text
    assert validate(agent.history) == []
    assert any("maximum" in message for _, message in ui.notices)
    assert agent.run_turn("continue").status == "completed"  # the session is still usable


def test_default_step_limit_is_30():
    assert AgentConfig().max_steps == 30 and AgentConfig().max_consecutive_failures == 5


def test_consecutive_tool_failures_stop_the_turn(workdir):
    script = [tool_turn(call("read_file", path="missing.txt")) for _ in range(5)]
    agent, provider, _ = make_agent(script, workdir, config=AgentConfig(max_consecutive_failures=3))
    result = agent.run_turn("read it")
    assert result.status == "too_many_failures" and provider.calls == 3
    assert agent.history[-1].role == "assistant" and "3 tool calls failed" in agent.history[-1].text
    assert validate(agent.history) == []


def test_a_success_resets_the_failure_counter(workdir):
    bad = tool_turn(call("read_file", path="missing.txt"))
    script = [bad, bad, tool_turn(call("list_files")), bad, bad, MockTurn(text="finished")]
    agent, _, _ = make_agent(script, workdir, config=AgentConfig(max_consecutive_failures=3))
    assert agent.run_turn("go").status == "completed"


def test_user_rejections_do_not_count_as_failures(workdir):
    ui = RecordingUI(default=False)
    script = [tool_turn(call("run_shell", command="echo hi")) for _ in range(3)] + [MockTurn(text="ok")]
    agent, _, _ = make_agent(script, workdir, ui=ui, config=AgentConfig(max_consecutive_failures=2))
    assert agent.run_turn("go").status == "completed"


# ----------------------------------------------------- provider errors


def test_fatal_error_is_reported_without_retry_and_the_session_survives(workdir):
    script = [MockTurn(error=FatalError("Authentication failed (401): check API key", 401)), MockTurn(text="back again")]
    agent, provider, ui = make_agent(script, workdir)
    result = agent.run_turn("hello")
    assert result.status == "error" and "401" in result.error
    assert provider.calls == 1 and agent.sleeps == []
    assert ui.notices[-1][0] == "error" and "401" in ui.notices[-1][1]
    second = agent.run_turn("hello again")
    assert second.status == "completed"
    assert validate(agent.history) == []
    assert [m.role for m in agent.history] == ["user", "assistant", "user", "assistant"]


def test_rate_limit_is_retried_with_backoff_and_shown(workdir):
    script = [
        MockTurn(error=RetryableError("Rate limited (429)", 429)),
        MockTurn(error=RetryableError("Rate limited (429)", 429)),
        MockTurn(text="ok"),
    ]
    agent, provider, ui = make_agent(script, workdir)
    result = agent.run_turn("go")
    assert result.status == "completed" and provider.calls == 3
    assert agent.sleeps == [1.0, 2.0]
    assert [r[0] for r in ui.retries] == [1, 2]


def test_retries_exhausted_ends_the_turn_with_a_clear_error(workdir):
    script = [MockTurn(error=RetryableError("Server error (503)", 503)) for _ in range(3)]
    agent, provider, _ = make_agent(script, workdir, max_retries=2)
    result = agent.run_turn("go")
    assert result.status == "error" and "Giving up after 3 attempts" in result.error
    assert provider.calls == 3


def test_unexpected_provider_exception_does_not_crash_the_agent(workdir):
    agent, _, ui = make_agent([MockTurn(error=RuntimeError("socket exploded"))], workdir)
    result = agent.run_turn("go")
    assert result.status == "error" and "RuntimeError" in result.error


def test_empty_model_response_is_retried_then_reported(workdir):
    agent, provider, ui = make_agent([MockTurn(), MockTurn(), MockTurn(text="finally")], workdir)
    result = agent.run_turn("go")
    assert result.status == "completed" and result.text == "finally" and provider.calls == 3
    assert [m.role for m in agent.history] == ["user", "assistant"]

    agent, provider, _ = make_agent([MockTurn(), MockTurn(), MockTurn()], workdir)
    result = agent.run_turn("go")
    assert result.status == "error" and "empty response" in result.error and provider.calls == 3
    assert validate(agent.history) == []


def test_missing_usage_falls_back_to_an_estimate(workdir):
    agent, _, ui = make_agent([MockTurn(text="a reply", omit_usage=True)], workdir)
    result = agent.run_turn("question")
    assert result.usage.estimated and result.usage.input_tokens > 0 and result.usage.output_tokens > 0
    assert ui.usages[0].estimated


# --------------------------------------------- malformed tool arguments


def test_truncated_arguments_are_never_executed(workdir):
    cut_off = MockToolCall("write_file", '{"path": "big.txt", "content": "abc')
    script = [MockTurn(tool_calls=[cut_off], stop_reason="max_tokens"), MockTurn(text="I will write it in pieces.")]
    agent, provider, _ = make_agent(script, workdir)
    result = agent.run_turn("write a big file")
    assert result.status == "completed"
    assert not (workdir / "big.txt").exists()
    stored = agent.history[1].tool_uses[0]
    assert stored.input == {} and stored.raw_arguments.startswith('{"path"')
    error = results_of(agent, 2)[0]
    assert error.is_error and "max_tokens" in error.content and "NOT executed" in error.content
    assert validate(agent.history) == []


def test_invalid_json_arguments_get_a_correctable_error(workdir):
    script = [tool_turn(MockToolCall("read_file", "{not json")), MockTurn(text="ok")]
    agent, _, _ = make_agent(script, workdir)
    agent.run_turn("go")
    error = results_of(agent, 2)[0]
    assert error.is_error and "not a valid JSON object" in error.content


# --------------------------------------------------- interruption (basic)


def test_interrupted_tool_round_still_leaves_a_valid_history(workdir):
    (workdir / "a.txt").write_text("A\n")

    class CancelOnConfirm(RecordingUI):
        def confirm(self, tool_name, preview):
            raise Cancelled()

    script = [
        tool_turn(call("read_file", path="a.txt"), call("write_file", path="b.txt", content="b"), call("list_files")),
        MockTurn(text="next turn works"),
    ]
    agent, provider, _ = make_agent(script, workdir, ui=CancelOnConfirm())
    result = agent.run_turn("go")
    assert result.status == "cancelled"
    results = results_of(agent, 2)
    assert [r.is_error for r in results] == [False, True, True]
    assert "NOT executed" in results[1].content and "NOT executed" in results[2].content
    assert not (workdir / "b.txt").exists()
    assert validate(agent.history) == []
    assert agent.run_turn("hello again").status == "completed"  # the next request is accepted
