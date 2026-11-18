"""4.3 malformed model output and loops: nothing may crash the agent, and the history must stay valid.

Already covered elsewhere (not repeated here): truncated or invalid JSON arguments are never executed
(test_agent_loop.py), empty responses (test_agent_loop.py), argument validation errors (test_tools.py).
"""

import sys

import pytest
from helpers import make_agent

from mini_agent.agent.compaction import collect_state
from mini_agent.agent.loop_detector import LoopDetector
from mini_agent.agent.text_toolcall import find_text_tool_calls
from mini_agent.config import AgentConfig
from mini_agent.history import validate
from mini_agent.messages import AGENT_NOTICE_PREFIX, Message, ToolUseBlock
from mini_agent.providers.mock import MockToolCall, MockTurn
from mini_agent.tools.registry import default_registry

NAMES = default_registry().names
PY = f'"{sys.executable}"'


def read_call(path="a.txt"):
    return MockToolCall("read_file", {"path": path})


def read_turn(path="a.txt"):
    return MockTurn(tool_calls=[read_call(path)])


# ---------------------------------------------- unknown tools and bad arguments, end to end


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (MockToolCall("read_fil", {"path": "a.txt"}), ["Unknown tool", "Did you mean 'read_file'"]),
        (MockToolCall("fly_to_moon", {}), ["Unknown tool", "Available tools"]),
        (MockToolCall("read_file", {}), ["missing required field 'path'"]),
        (MockToolCall("read_file", {"path": "a.txt", "start_line": "first"}), ["start_line"]),
        (MockToolCall("read_file", {"path": ["a.txt"]}), ["path"]),
    ],
)
def test_bad_tool_calls_become_fixable_errors_and_the_task_continues(workdir, call, expected):
    (workdir / "a.txt").write_text("hello\n")
    agent, provider, ui = make_agent([MockTurn(tool_calls=[call]), read_turn(), MockTurn(text="done")], workdir)
    outcome = agent.run_turn("go")
    assert outcome.status == "completed", outcome.error
    first = agent.history[2].tool_results[0]
    assert first.is_error and all(part in first.content for part in expected)
    assert "hello" in agent.history[4].tool_results[0].content  # the corrected call worked
    assert validate(agent.history) == []


# ------------------------------------------------------- tool calls written as text


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "read_file", "arguments": {"path": "a.txt"}}',
        'I will read it.\n```json\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n```',
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>',
        '{"tool": "read_file", "parameters": {"path": "a.txt"}}',
        '{"type": "function", "function": {"name": "read_file", "arguments": "{\\"path\\": \\"a.txt\\"}"}}',
        'First {"name": "list_files", "arguments": {}} then {"name": "read_file", "arguments": {"path": "a.txt"}}',
    ],
)
def test_tool_calls_written_as_text_are_recognized(text):
    calls = find_text_tool_calls(text, NAMES)
    assert any(c.name == "read_file" and c.arguments == {"path": "a.txt"} for c in calls)
    assert all(isinstance(c.arguments, dict) for c in calls)


def test_the_arguments_are_read_out_of_the_text():
    [call] = find_text_tool_calls('{"function": {"name": "grep", "arguments": "{\\"pattern\\": \\"x\\"}"}}', NAMES)
    assert call.name == "grep" and call.arguments == {"pattern": "x"}


@pytest.mark.parametrize(
    "text",
    [
        "Here is the plan. First read the file, then edit it.",
        '{"name": "fly_to_moon", "arguments": {}}',  # not a registered tool
        '{"name": "read_file"}',  # no arguments field: probably just data
        '{"name": "my-package", "version": "1.0.0"}',  # an ordinary package.json
        '{"name": "read_file", "arguments": {"path": ',  # truncated JSON
        '{"name": "read_file", "arguments": "not json"}',
        '{"name": "read_file", "arguments": [1, 2]}',
        "{" * 5_000,
        "",
    ],
)
def test_ordinary_text_and_data_are_left_alone(text):
    assert find_text_tool_calls(text, NAMES) == []


def test_a_text_tool_call_is_not_executed_and_the_model_is_asked_to_use_the_tool_interface(workdir):
    written = '{"name": "write_file", "arguments": {"path": "evil.txt", "content": "x"}}'
    script = [MockTurn(text=written), read_turn(), MockTurn(text="read it")]
    (workdir / "a.txt").write_text("hello\n")
    agent, provider, ui = make_agent(script, workdir, config=AgentConfig(auto_yes=True))
    outcome = agent.run_turn("go")
    assert outcome.status == "completed", outcome.error
    assert not (workdir / "evil.txt").exists()
    notice = agent.history[2]
    assert notice.role == "user" and notice.text.startswith(AGENT_NOTICE_PREFIX) and "write_file" in notice.text
    assert provider.requests[1].messages[-1] == notice  # the model saw the correction
    assert [m.role for m in agent.history] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert validate(agent.history) == []
    assert any("wrote a tool call as text" in message for _, message in ui.notices)


def test_the_agent_gives_up_after_the_allowed_number_of_corrections(workdir):
    written = '{"name": "run_shell", "arguments": {"command": "echo hi"}}'
    agent, provider, ui = make_agent([MockTurn(text=written)] * 3, workdir, config=AgentConfig(auto_yes=True))
    outcome = agent.run_turn("go")
    assert outcome.status == "malformed_output" and provider.calls == 3
    assert agent.history[-1].role == "assistant" and validate(agent.history) == []
    assert not any(m.tool_results for m in agent.history)  # nothing was executed


def test_the_number_of_corrections_is_configurable(workdir):
    written = '{"name": "run_shell", "arguments": {"command": "echo hi"}}'
    agent, provider, ui = make_agent([MockTurn(text=written)], workdir, config=AgentConfig(max_text_tool_corrections=0))
    assert agent.run_turn("go").status == "malformed_output" and provider.calls == 1


def test_agent_notices_are_not_mistaken_for_user_requests_by_compaction(workdir):
    written = '{"name": "read_file", "arguments": {"path": "a.txt"}}'
    agent, provider, ui = make_agent([MockTurn(text=written), MockTurn(text="ok")], workdir)
    agent.run_turn("Please summarize the project")
    requests, _ = collect_state(agent.history)
    assert requests == ["Please summarize the project"]


# ------------------------------------------------ stopped replies and unknown stop reasons


def test_a_reply_stopped_by_the_provider_never_runs_its_tool_calls(workdir):
    script = [MockTurn(tool_calls=[MockToolCall("write_file", {"path": "x.txt", "content": "x"})], stop_reason="error"), MockTurn(text="ok")]
    agent, provider, ui = make_agent(script, workdir, config=AgentConfig(auto_yes=True))
    outcome = agent.run_turn("go")
    assert outcome.status == "completed" and not (workdir / "x.txt").exists()
    assert agent.history[2].tool_results[0].is_error and "NOT executed" in agent.history[2].tool_results[0].content
    assert validate(agent.history) == []


def test_a_text_reply_stopped_by_the_provider_is_shown_with_a_warning(workdir):
    agent, provider, ui = make_agent([MockTurn(text="I cannot", stop_reason="error")], workdir)
    outcome = agent.run_turn("go")
    assert outcome.status == "completed" and outcome.text == "I cannot"
    assert any("stopped this reply" in message for _, message in ui.notices)


# ----------------------------------------------------------------- loop detector (unit)


def use(name="read_file", **args):
    return ToolUseBlock(id="t", name=name, input=args)


def feed(detector, calls):
    """Feed (use, result) pairs and return the list of verdict levels (0 for none)."""
    return [(v.level if (v := detector.record(u, r, False)) else 0) for u, r in calls]


A = (use(path="a.txt"), "same content")
B = (use(path="b.txt"), "other content")
C = (use("run_shell", command="pytest"), "3 failed")


def test_the_same_call_with_the_same_result_escalates_hint_strong_hint_stop():
    assert feed(LoopDetector(), [A] * 6) == [0, 0, 1, 2, 3, 3]


def test_the_same_call_with_a_different_result_is_progress_not_a_loop():
    calls = [(use("run_shell", command="pytest"), f"{n} failed") for n in range(8)]
    assert feed(LoopDetector(), calls) == [0] * 8


def test_argument_order_and_spacing_do_not_matter():
    first = (ToolUseBlock(id="1", name="grep", input={"pattern": "x", "path": "src"}), "r")
    second = (ToolUseBlock(id="2", name="grep", input={"path": "src", "pattern": "x"}), "r")
    assert feed(LoopDetector(), [first, second, first]) == [0, 0, 1]


def test_an_error_and_a_success_with_the_same_text_are_different_outcomes():
    detector = LoopDetector()
    results = [detector.record(use(path="a"), "same", is_error) for is_error in (True, False, True)]
    assert results == [None, None, None]


def test_a_two_step_cycle_is_detected_after_three_rounds():
    assert feed(LoopDetector(), [A, B] * 3) == [0, 0, 0, 0, 0, 1]
    assert feed(LoopDetector(), [A, B] * 2 + [A]) == [0] * 5


def test_a_three_step_cycle_is_detected_after_three_rounds():
    assert feed(LoopDetector(), [A, B, C] * 3) == [0] * 8 + [1]


def test_a_cycle_keeps_escalating_while_it_continues():
    assert feed(LoopDetector(), [A, B] * 5) == [0, 0, 0, 0, 0, 1, 2, 3, 3, 3]


def test_breaking_the_pattern_resets_the_escalation():
    calls = [A, A, A, A, B, A, A, A]
    assert feed(LoopDetector(), calls) == [0, 0, 1, 2, 0, 0, 0, 1]


def test_a_cycle_whose_results_change_is_not_a_loop():
    calls = [(use(path="a.txt"), f"v{n}") if n % 2 == 0 else B for n in range(8)]
    assert feed(LoopDetector(), calls) == [0] * 8


def test_thresholds_are_configurable():
    assert feed(LoopDetector(threshold=2), [A, A]) == [0, 1]
    assert feed(LoopDetector(threshold=3, max_period=2), [A, B, C] * 3) == [0] * 9


def test_reset_forgets_everything():
    detector = LoopDetector()
    feed(detector, [A, A])
    detector.reset()
    assert feed(detector, [A]) == [0]


def test_the_verdict_names_what_is_repeating():
    detector = LoopDetector()
    verdicts = [detector.record(u, r, False) for u, r in [A, B] * 3]
    verdict = verdicts[-1]
    assert verdict.rule == "cycle" and "read_file a.txt -> read_file b.txt" in verdict.description
    assert "read_file a.txt" in feed_verdict(A, 3).hint()


def feed_verdict(pair, times):
    detector = LoopDetector()
    verdict = None
    for _ in range(times):
        verdict = detector.record(pair[0], pair[1], False)
    return verdict


# ------------------------------------------------------------- loop detector (in the agent)


def test_a_repeated_read_gets_a_hint_in_its_tool_result_and_the_task_can_still_finish(workdir):
    (workdir / "a.txt").write_text("hello\n")
    script = [read_turn(), read_turn(), read_turn(), MockTurn(text="I will stop reading it.")]
    agent, provider, ui = make_agent(script, workdir)
    outcome = agent.run_turn("go")
    assert outcome.status == "completed", outcome.error
    results = [m.tool_results[0].content for m in agent.history if m.tool_results]
    assert "Loop warning" not in results[0] and "Loop warning" not in results[1]
    assert "Loop warning" in results[2] and "hello" in results[2]  # the real result is still there
    assert len([m for m in agent.history if m.role == "user"]) == 4  # the hint added no extra message
    assert validate(agent.history) == []


def test_a_model_that_ignores_the_warnings_is_stopped_with_a_question_to_the_user(workdir):
    (workdir / "a.txt").write_text("hello\n")
    agent, provider, ui = make_agent([read_turn()] * 5, workdir)
    outcome = agent.run_turn("go")
    assert outcome.status == "loop_detected" and provider.calls == 5
    results = [m.tool_results[0].content for m in agent.history if m.tool_results]
    assert "STRONG LOOP WARNING" in results[3]
    last = agent.history[-1]
    assert last.role == "assistant" and "Tell me how you would like to proceed" in last.text
    assert validate(agent.history) == []


def test_a_cycle_between_two_files_is_detected(workdir):
    (workdir / "a.txt").write_text("A\n")
    (workdir / "b.txt").write_text("B\n")
    script = [read_turn("a.txt"), read_turn("b.txt")] * 3 + [MockTurn(text="ok")]
    agent, provider, ui = make_agent(script, workdir)
    assert agent.run_turn("go").status == "completed"
    results = [m.tool_results[0].content for m in agent.history if m.tool_results]
    assert "Loop warning" in results[5] and "circle" in results[5]


def test_the_same_command_with_changing_output_is_not_flagged(workdir):
    counter = (
        f"{PY} -c \"import pathlib; p = pathlib.Path('n.txt'); "
        "n = int(p.read_text()) + 1 if p.exists() else 1; p.write_text(str(n)); print(n)\""
    )
    script = [MockTurn(tool_calls=[MockToolCall("run_shell", {"command": counter})]) for _ in range(5)] + [MockTurn(text="ok")]
    agent, provider, ui = make_agent(script, workdir, config=AgentConfig(auto_yes=True))
    outcome = agent.run_turn("go")
    assert outcome.status == "completed", outcome.error
    assert not any("Loop warning" in b.content for m in agent.history for b in m.tool_results)


def test_identical_calls_inside_one_reply_are_stopped_part_way_and_every_call_gets_a_result(workdir):
    (workdir / "a.txt").write_text("hello\n")
    agent, provider, ui = make_agent([MockTurn(tool_calls=[read_call()] * 6)], workdir)
    outcome = agent.run_turn("go")
    assert outcome.status == "loop_detected" and provider.calls == 1
    results = agent.history[2].tool_results
    assert len(results) == 6 and results[5].is_error and "NOT executed" in results[5].content
    assert validate(agent.history) == []


def test_the_next_task_starts_with_a_clean_slate(workdir):
    (workdir / "a.txt").write_text("hello\n")
    agent, provider, ui = make_agent([read_turn()] * 5 + [read_turn(), MockTurn(text="fine")], workdir)
    assert agent.run_turn("first").status == "loop_detected"
    assert agent.run_turn("second").status == "completed"
    last_results = agent.history[-2].tool_results
    assert "Loop warning" not in last_results[0].content


def test_loop_detection_events_are_logged(workdir, tmp_path):
    from mini_agent.logging_ import Redactor, RequestLogger

    (workdir / "a.txt").write_text("hello\n")
    log_path = tmp_path / "run.log"
    logger = RequestLogger(str(log_path), Redactor())
    agent, provider, ui = make_agent([read_turn()] * 3 + [MockTurn(text="ok")], workdir, logger=logger)
    agent.run_turn("go")
    assert "loop_detected" in log_path.read_text(encoding="utf-8")
