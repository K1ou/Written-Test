"""4.1 context overflow: compaction must keep the history valid and keep what the task depends on."""

import re

import pytest
from helpers import make_agent

import mini_agent.agent.compaction as compaction_module
from mini_agent.agent.cancel import CancelToken
from mini_agent.agent.compaction import (
    CompactionError,
    OMITTED_OUTPUT,
    SUMMARY_PREFIX,
    compact,
    parse_summary,
    shrink_message,
    split_units,
)
from mini_agent.agent.loop import SUMMARY_SYSTEM
from mini_agent.config import AgentConfig
from mini_agent.history import validate
from mini_agent.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mini_agent.providers.base import Done, TextDelta, UsageEvent
from mini_agent.providers.errors import ContextLengthExceeded, FatalError
from mini_agent.providers.mock import MockProvider, MockToolCall, MockTurn
from mini_agent.tokens import estimate_message_tokens, estimate_request_tokens

# ------------------------------------------------------------------ builders


def tool_use(id, name="read_file", **args):
    return ToolUseBlock(id=id, name=name, input=args)


def result(id, content, is_error=False):
    return ToolResultBlock(tool_use_id=id, content=content, is_error=is_error)


def group(n, *, calls=1, size=1_000, name="read_file", path=None):
    """One assistant message with `calls` tool calls plus the user message holding all their results."""
    ids = [f"t{n}_{i}" for i in range(calls)]
    uses = [tool_use(i, name, path=path or f"file{n}_{k}.txt") for k, i in enumerate(ids)]
    results = [result(i, f"content of {i} " + "x" * size) for i in ids]
    return [Message(role="assistant", content=uses), Message(role="user", content=results)]


def history(groups=8, *, size=1_000, calls=1):
    messages = [Message.user("Please fix the failing tests in the project.")]
    for n in range(groups):
        messages += group(n, calls=calls, size=size)
    messages.append(Message.assistant("All done."))
    return messages


def tokens(messages):
    return sum(estimate_message_tokens(m) for m in messages)


# --------------------------------------------------------------------- units


def test_a_tool_call_group_with_three_calls_is_one_unit():
    messages = [Message.user("go"), *group(1, calls=3), Message.assistant("ok")]
    units = split_units(messages)
    assert [len(u) for u in units] == [1, 2, 1]
    assert len(units[1][0].tool_uses) == 3 and len(units[1][1].tool_results) == 3


def test_splitting_refuses_a_call_without_results():
    with pytest.raises(CompactionError):
        split_units([Message.user("go"), group(1)[0]])


# ----------------------------------------------------- validity under every cut


def _invented_results(messages, original):
    """Tool results that are neither original nor a shrink placeholder, i.e. made up by repair()."""
    known = {r.content for m in original for r in m.tool_results}
    return [
        r.content
        for m in messages
        for r in m.tool_results
        if r.content not in known and not r.content.startswith(OMITTED_OUTPUT)
    ]


def _sweep_problems(original, keep):
    problems = []
    for target in range(0, tokens(original) + 200, 37):
        try:
            out = compact(original, target_tokens=target, keep_recent_units=keep)
        except CompactionError:
            problems.append((target, "error"))
            continue
        if validate(out.messages):
            problems.append((target, "invalid"))
        elif "history needed repair" in out.notes or _invented_results(out.messages, original):
            problems.append((target, "repair was needed"))
    return problems


@pytest.mark.parametrize("calls", [1, 3])
@pytest.mark.parametrize("keep", [1, 2, 3])
def test_every_budget_gives_a_valid_history_that_keeps_the_latest_unit(calls, keep):
    original = history(8, size=800, calls=calls)
    assert _sweep_problems(original, keep) == []
    for target in range(0, tokens(original) + 200, 37):
        out = compact(original, target_tokens=target, keep_recent_units=keep)
        assert out.messages[0].role == "user"
        if out.changed:
            assert out.messages[-1] == original[-1]


def test_the_sweep_fails_when_units_are_split_per_message(monkeypatch):
    """Mutation check done in memory: with a broken split_units the sweep must notice."""
    monkeypatch.setattr(compaction_module, "split_units", lambda messages: [[m] for m in messages])
    assert _sweep_problems(history(8, size=800, calls=3), keep=1) != []


def test_a_naive_message_count_cut_would_produce_invalid_histories():
    """Negative control: the failure that atomic units prevent really does happen with plain slicing."""
    original = history(5, calls=2)
    broken = [k for k in range(1, len(original)) if validate(original[k:])]
    assert broken, "slicing never broke the history, so the sweep above proves nothing"
    assert len(broken) > len(original) // 2


def test_invalid_input_is_rejected_instead_of_made_worse():
    broken = [Message.user("go"), group(1)[0], Message.assistant("x")]
    with pytest.raises(CompactionError):
        compact(broken, target_tokens=10)


# ---------------------------------------------------------- layer 1 and layer 2


def test_shrinking_old_tool_output_alone_is_used_when_it_is_enough():
    original = history(6, size=3_000)
    out = compact(original, target_tokens=tokens(original) // 3, keep_recent_units=2)
    assert out.layer == 1 and out.shrunk_units > 0
    assert len(out.messages) == len(original)  # no message removed, so no information about order lost
    ids_before = [b.id for m in original for b in m.tool_uses]
    ids_after = [b.id for m in out.messages for b in m.tool_uses]
    assert ids_before == ids_after
    shrunk = out.messages[2].tool_results[0]
    assert shrunk.content.startswith(OMITTED_OUTPUT) and shrunk.tool_use_id == original[2].tool_results[0].tool_use_id
    # the newest units stay verbatim
    assert out.messages[-3:] == original[-3:]


def test_shrinking_keeps_error_flags_and_short_results():
    message = Message(role="user", content=[result("a", "short"), result("b", "E" * 500, is_error=True)])
    shrunk = shrink_message(message)
    assert shrunk.tool_results[0].content == "short"
    assert shrunk.tool_results[1].is_error and shrunk.tool_results[1].content.startswith(OMITTED_OUTPUT)


def test_big_tool_arguments_are_shortened_but_the_call_stays():
    big = Message(role="assistant", content=[tool_use("w", "write_file", path="a.py", content="z" * 5_000)])
    shrunk = shrink_message(big).tool_uses[0]
    assert shrunk.id == "w" and shrunk.input["path"] == "a.py" and "5000" in shrunk.input["content"]


def test_summarizing_replaces_old_units_with_one_user_message():
    original = history(10, size=2_000)
    out = compact(original, target_tokens=300, keep_recent_units=2)
    assert out.layer == 2 and out.summarized_messages > 0
    first = out.messages[0]
    assert first.role == "user" and first.text.startswith(SUMMARY_PREFIX)
    assert "Please fix the failing tests" in first.text
    assert len(out.messages) < len(original)
    assert out.messages[-1] == original[-1]


def test_summary_text_from_the_model_is_included_and_flagged():
    original = history(10, size=2_000)
    out = compact(original, target_tokens=300, summarize=lambda prefix: "Goal: tests\nUnfinished: lint")
    assert out.llm_summary and "Unfinished: lint" in out.messages[0].text


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_without_a_model_summary_rule_based_notes_warn_about_unfinished_work(answer):
    original = history(10, size=2_000)
    out = compact(original, target_tokens=300, summarize=lambda prefix: answer)
    assert not out.llm_summary
    assert "Unfinished work" in out.messages[0].text and "tool call(s)" in out.messages[0].text


def test_nothing_is_done_when_it_already_fits_or_there_is_too_little_to_cut():
    original = history(3, size=100)
    assert compact(original, target_tokens=10**6).layer == 0
    tiny = [Message.user("hi"), Message.assistant("hello")]
    out = compact(tiny, target_tokens=1, keep_recent_units=2)
    assert out.layer == 0 and out.messages == tiny


# --------------------------------------------------- what survives compaction


def _task_history():
    messages = [Message.user("Request ALPHA: create the module")]
    messages += [
        Message(role="assistant", content=[tool_use("w1", "write_file", path="src/a.py", content="a" * 600)]),
        Message(role="user", content=[result("w1", "Created src/a.py " + "y" * 300)]),
        Message(role="assistant", content=[tool_use("w2", "edit_file", path="src/failed.py", old_text="a", new_text="b")]),
        Message(role="user", content=[result("w2", "No match " + "n" * 300, is_error=True)]),
        Message.assistant("Module created."),
        Message.user("Request BETA: now add tests"),
    ]
    for n in range(6):
        messages += group(n, size=1_500)
    messages.append(Message.assistant("Tests added."))
    return messages


def test_requests_and_modified_files_survive_two_compactions():
    first = compact(_task_history(), target_tokens=300, keep_recent_units=1)
    assert first.layer == 2
    requests, files = parse_summary(first.messages[0].text)
    assert [r.split(":")[0] for r in requests] == ["Request ALPHA", "Request BETA"]
    assert files == ["src/a.py"]  # the failed edit is not a modification

    more = list(first.messages) + [
        Message.user("Request GAMMA: update the docs"),
        Message(role="assistant", content=[tool_use("w3", "write_file", path="docs/readme.md", content="d" * 600)]),
        Message(role="user", content=[result("w3", "Created docs/readme.md " + "y" * 300)]),
        *group(50, size=2_000),
        Message.assistant("Docs updated."),
    ]
    second = compact(more, target_tokens=1, keep_recent_units=1)
    assert second.layer == 2
    assert sum(1 for m in second.messages if m.text.startswith(SUMMARY_PREFIX)) == 1
    requests, files = parse_summary(second.messages[0].text)
    assert [r.split(":")[0] for r in requests] == ["Request ALPHA", "Request BETA", "Request GAMMA"]
    assert files == ["src/a.py", "docs/readme.md"]
    assert validate(second.messages) == []


def test_the_summary_merges_with_a_following_user_message():
    original = history(6, size=2_000)
    original[-1:] = [Message.assistant("ok"), Message.user("and now the next thing")]
    out = compact(original, target_tokens=300, keep_recent_units=1)
    assert validate(out.messages) == []
    assert out.messages[-1].text.endswith("and now the next thing") or out.messages[-1].text == "and now the next thing"
    assert sum(1 for m in out.messages if m.role == "user" and m.text.startswith(SUMMARY_PREFIX)) <= 1


# ------------------------------------------------------------------- the agent


class SummaryMock(MockProvider):
    """Answers summary requests itself and sends everything else to the scripted turns."""

    def __init__(self, script, summary="Goal: finish\nDone: reading\nUnfinished: the last file\nKey facts: none", fail=None):
        super().__init__(script)
        self.summary = summary
        self.fail = fail
        self.summary_requests = []

    def stream(self, request, cancel):
        if request.system != SUMMARY_SYSTEM:
            yield from super().stream(request, cancel)
            return
        self.summary_requests.append(request)
        if self.fail is not None:
            raise self.fail
        yield TextDelta(self.summary)
        yield UsageEvent(Usage(input_tokens=100, output_tokens=20))
        yield Done("end_turn")


def read_turn(i):
    return MockTurn(tool_calls=[MockToolCall("read_file", {"path": f"f{i}.txt"})])


def put_files(workdir, count=12, lines=100):
    for i in range(count):
        (workdir / f"f{i}.txt").write_text("".join(f"line {n} of file {i} " + "z" * 40 + "\n" for n in range(lines)))


SMALL = dict(context_window=8_000, compact_threshold=0.8, compact_target=0.5)


def run_reading(workdir, provider, **overrides):
    put_files(workdir)
    config = AgentConfig(**{**SMALL, **overrides})
    agent, provider, ui = make_agent([], workdir, config=config, provider=provider)
    return agent, provider, ui, agent.run_turn("Read every file and tell me when you are done.")


def reading_script(count=12):
    return [*[read_turn(i) for i in range(count)], MockTurn(text="Read them all.")]


def test_a_long_task_never_sends_more_than_the_window_and_never_an_invalid_history(workdir):
    provider = MockProvider(reading_script())
    agent, provider, ui, outcome = run_reading(workdir, provider, llm_summary=False)
    assert outcome.status == "completed", outcome.error
    assert any("Context compacted" in message for _, message in ui.notices)
    for request in provider.requests:
        assert estimate_request_tokens(request.system, request.messages, request.tools) <= 8_000
    # the strict mock already rejected any invalid request; the final history is valid too
    assert validate(agent.history) == []
    first = provider.requests[-1].messages[0]
    assert first.text.startswith(SUMMARY_PREFIX) and "Read every file" in first.text


def test_the_model_written_summary_is_requested_once_per_compaction_and_billed(workdir):
    provider = SummaryMock(reading_script())
    agent, provider, ui, outcome = run_reading(workdir, provider)
    assert outcome.status == "completed", outcome.error
    assert provider.summary_requests, "no summary was requested"
    request = provider.summary_requests[0]
    assert request.tools == [] and [m.role for m in request.messages] == ["user"]
    assert "Unfinished: the last file" in provider.requests[-1].messages[0].text
    assert outcome.usage.input_tokens >= 100 * len(provider.summary_requests)


def test_a_failing_summary_call_falls_back_to_rule_based_notes(workdir):
    provider = SummaryMock(reading_script(), fail=FatalError("summary backend down", 500))
    agent, provider, ui, outcome = run_reading(workdir, provider)
    assert outcome.status == "completed", outcome.error
    assert "Unfinished work" in provider.requests[-1].messages[0].text


def test_cancelling_during_the_summary_call_leaves_the_history_untouched_and_valid(workdir):
    from mini_agent.agent.cancel import Cancelled

    provider = SummaryMock(reading_script(), fail=Cancelled())
    agent, provider, ui, outcome = run_reading(workdir, provider)
    assert outcome.status == "cancelled"
    assert validate(agent.history) == []
    assert not any(m.text.startswith(SUMMARY_PREFIX) for m in agent.history)


def test_compaction_is_not_repeated_on_every_step_when_it_cannot_get_below_the_threshold(workdir):
    # one result is bigger than the whole window and is kept verbatim as the newest unit
    messages = [Message.user("start"), *group(1, size=40_000), Message.assistant("ok"), Message.user("next")]
    agent, provider, ui = make_agent([], workdir, config=AgentConfig(llm_summary=False, **SMALL))
    agent.history[:] = [*messages[:1], *group(0, size=500), *group(1, size=40_000)]
    cancel = CancelToken()
    agent._maybe_compact(1, cancel, force=False)
    notices = len(ui.notices)
    snapshot = list(agent.history)
    agent._maybe_compact(2, cancel, force=False)
    assert agent.history == snapshot and len(ui.notices) == notices


def test_calibration_uses_real_usage_but_ignores_implausible_numbers(workdir):
    agent, provider, ui = make_agent([], workdir)
    messages = history(4)
    request = agent._build_request()
    request.messages[:] = messages
    raw = estimate_request_tokens(request.system, request.messages, request.tools)
    agent._calibrate(request, Usage(input_tokens=10, output_tokens=1))  # the mock's fake usage
    assert agent._token_scale == 1.0
    agent._calibrate(request, Usage(input_tokens=raw * 2, output_tokens=1))
    assert agent._token_scale == pytest.approx(2.0)
    agent._calibrate(request, Usage(input_tokens=raw * 100, output_tokens=1))
    assert agent._token_scale == pytest.approx(2.0)


# ---------------------------------------------- the provider says "too long" anyway


def overflow():
    return ContextLengthExceeded("This model's maximum context length is exceeded", 400)


def loaded_agent(workdir, script, **config):
    agent, provider, ui = make_agent(script, workdir, config=AgentConfig(llm_summary=False, **config))
    agent.history[:] = history(10, size=3_000)[:-1] + [Message.assistant("Finished the reading.")]
    return agent, provider, ui


def test_context_length_error_triggers_one_forced_compaction_and_a_retry(workdir):
    agent, provider, ui = loaded_agent(workdir, [MockTurn(error=overflow()), MockTurn(text="Carrying on.")])
    outcome = agent.run_turn("one more thing")
    assert outcome.status == "completed" and outcome.text == "Carrying on."
    assert provider.calls == 2
    first, second = provider.requests
    assert len(second.messages) <= len(first.messages)
    assert estimate_request_tokens(second.system, second.messages) < estimate_request_tokens(first.system, first.messages)
    assert any("Context compacted" in message for _, message in ui.notices)
    assert validate(agent.history) == []


def test_a_second_context_length_error_is_reported_not_retried_forever(workdir):
    agent, provider, ui = loaded_agent(workdir, [MockTurn(error=overflow()), MockTurn(error=overflow())])
    outcome = agent.run_turn("one more thing")
    assert outcome.status == "error" and "context length" in outcome.error
    assert provider.calls == 2


def test_overflow_with_nothing_to_compact_is_reported_at_once(workdir):
    agent, provider, ui = make_agent([MockTurn(error=overflow())], workdir)
    outcome = agent.run_turn("hi")
    assert outcome.status == "error" and provider.calls == 1


# --------------------------------------------------------- oversized tool output


def test_a_five_megabyte_log_cannot_flood_the_context(workdir):
    (workdir / "app.log").write_text("".join(f"2024-01-01 line {n} " + "e" * 40 + "\n" for n in range(100_000)))
    script = [
        MockTurn(tool_calls=[MockToolCall("read_file", {"path": "app.log"})]),
        MockTurn(tool_calls=[MockToolCall("run_shell", {"command": "python -c \"print('o' * 5000000)\""})]),
        MockTurn(text="Looked at it."),
    ]
    agent, provider, ui = make_agent(script, workdir, config=AgentConfig(auto_yes=True))
    outcome = agent.run_turn("what is wrong in app.log?")
    assert outcome.status == "completed", outcome.error
    read, shell = (agent.history[2].tool_results[0], agent.history[4].tool_results[0])
    assert len(read.content) <= 30_000 and re.search(r"start_line=\d+ to continue", read.content)
    assert len(shell.content) <= 30_000 and "omitted" in shell.content
    for request in provider.requests:
        assert estimate_request_tokens(request.system, request.messages, request.tools) < 20_000
