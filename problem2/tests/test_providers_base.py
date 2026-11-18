import threading

import pytest

from mini_agent.agent.cancel import CancelToken, Cancelled
from mini_agent.messages import Message, TextBlock, ToolUseBlock
from mini_agent.providers.base import (
    Done,
    ProviderRequest,
    StreamAccumulator,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
    collect,
)
from mini_agent.providers.errors import (
    ContextLengthExceeded,
    FatalError,
    RetryableError,
    from_http_status,
    parse_retry_after,
)
from mini_agent.providers.mock import MockProvider, MockToolCall, MockTurn
from mini_agent.messages import ToolResultBlock, Usage


def request(*messages: Message) -> ProviderRequest:
    return ProviderRequest(system="sys", messages=list(messages) or [Message.user("hi")])


def run(provider: MockProvider, req: ProviderRequest, cancel: CancelToken | None = None):
    cancel = cancel or CancelToken()
    return collect(provider.stream(req, cancel), cancel)


# ---------- accumulator ----------


def test_accumulator_joins_text_and_fragmented_tool_arguments():
    acc = StreamAccumulator()
    for ev in [
        TextDelta("Let me "),
        TextDelta("look."),
        ToolCallStart("t1", "read_file"),
        ToolCallArgsDelta("t1", '{"path": "a'),
        ToolCallArgsDelta("t1", '.py"}'),
        ToolCallEnd("t1"),
        UsageEvent(Usage(input_tokens=3, output_tokens=4)),
        Done("tool_use"),
    ]:
        acc.feed(ev)
    result = acc.result()
    assert result.stop_reason == "tool_use"
    assert result.usage == Usage(input_tokens=3, output_tokens=4)
    assert isinstance(result.message.content[0], TextBlock)
    assert result.message.text == "Let me look."
    assert result.message.tool_uses[0].input == {"path": "a.py"}


def test_accumulator_keeps_interleaved_parallel_tool_calls_separate():
    acc = StreamAccumulator()
    for ev in [
        ToolCallStart("a", "read_file"),
        ToolCallStart("b", "grep"),
        ToolCallArgsDelta("a", '{"path":'),
        ToolCallArgsDelta("b", '{"pattern": "x"}'),
        ToolCallArgsDelta("a", ' "f"}'),
        ToolCallEnd("a"),
        ToolCallEnd("b"),
        Done("tool_use"),
    ]:
        acc.feed(ev)
    uses = acc.result().message.tool_uses
    assert [(u.id, u.input) for u in uses] == [("a", {"path": "f"}), ("b", {"pattern": "x"})]


def test_accumulator_truncated_arguments_are_flagged_not_repaired():
    acc = StreamAccumulator()
    for ev in [
        ToolCallStart("a", "write_file"),
        ToolCallArgsDelta("a", '{"path": "x", "content": "abc'),
        ToolCallEnd("a"),
        Done("max_tokens"),
    ]:
        acc.feed(ev)
    result = acc.result()
    use = result.message.tool_uses[0]
    assert result.stop_reason == "max_tokens"
    assert use.input == {} and use.raw_arguments is not None and not use.arguments_valid


def test_accumulator_vendor_meta_is_carried_to_the_tool_use():
    acc = StreamAccumulator()
    for ev in [ToolCallStart("a", "grep", meta={"sig": "xyz"}), ToolCallEnd("a"), Done("tool_use")]:
        acc.feed(ev)
    assert acc.result().message.tool_uses[0].meta == {"sig": "xyz"}


def test_partial_message_drops_incomplete_tool_calls_but_keeps_text_and_finished_calls():
    acc = StreamAccumulator()
    for ev in [
        TextDelta("working"),
        ToolCallStart("a", "read_file"),
        ToolCallArgsDelta("a", '{"path": "x"}'),
        ToolCallEnd("a"),
        ToolCallStart("b", "write_file"),
        ToolCallArgsDelta("b", '{"path": "y", "con'),
    ]:
        acc.feed(ev)
    partial = acc.partial_message()
    assert partial is not None
    assert partial.text == "working"
    assert [u.id for u in partial.tool_uses] == ["a"]


def test_partial_message_is_none_when_nothing_complete():
    acc = StreamAccumulator()
    acc.feed(ToolCallStart("a", "write_file"))
    acc.feed(ToolCallArgsDelta("a", '{"pa'))
    assert acc.partial_message() is None


def test_stream_without_done_is_a_retryable_dropped_connection():
    acc = StreamAccumulator()
    acc.feed(TextDelta("hel"))
    with pytest.raises(RetryableError):
        acc.result()


# ---------- mock provider ----------


def test_mock_streams_text_and_tool_calls_in_small_chunks():
    provider = MockProvider(
        [MockTurn(text="hello world, this is long", tool_calls=[MockToolCall("read_file", {"path": "some/long/path.py"})])],
        chunk_size=4,
    )
    events = []
    cancel = CancelToken()
    result = collect(provider.stream(request(), cancel), cancel, on_event=events.append)
    assert sum(isinstance(e, TextDelta) for e in events) > 3
    assert sum(isinstance(e, ToolCallArgsDelta) for e in events) > 3
    assert result.message.text == "hello world, this is long"
    assert result.message.tool_uses[0].input == {"path": "some/long/path.py"}
    assert result.stop_reason == "tool_use"


def test_mock_default_tool_call_ids_are_unique_across_turns():
    provider = MockProvider(
        [MockTurn(tool_calls=[MockToolCall("a"), MockToolCall("b")]), MockTurn(tool_calls=[MockToolCall("c")])]
    )
    first = run(provider, request())
    second = run(provider, request(Message.user("x")))
    ids = [u.id for u in first.message.tool_uses + second.message.tool_uses]
    assert len(set(ids)) == 3


def test_mock_rejects_invalid_history_with_400():
    provider = MockProvider([MockTurn(text="x")])
    bad = request(
        Message.user("hi"),
        Message(role="assistant", content=[ToolUseBlock(id="a", name="read_file")]),
        Message.user("no results here"),
    )
    with pytest.raises(FatalError) as info:
        run(provider, bad)
    assert info.value.status_code == 400 and "I1" in info.value.message


def test_mock_rejects_history_not_ending_with_user_message():
    provider = MockProvider([MockTurn(text="x")])
    with pytest.raises(FatalError):
        run(provider, request(Message.user("hi"), Message.assistant("hello")))


def test_mock_accepts_valid_tool_roundtrip_history():
    provider = MockProvider([MockTurn(text="done")])
    history = request(
        Message.user("hi"),
        Message(role="assistant", content=[ToolUseBlock(id="a", name="read_file")]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="a", content="data")]),
    )
    assert run(provider, history).message.text == "done"


def test_mock_records_requests_as_snapshots():
    provider = MockProvider([MockTurn(text="x")])
    msgs = [Message.user("hi")]
    run(provider, ProviderRequest(system="s", messages=msgs))
    msgs.append(Message.assistant("later"))
    assert len(provider.requests[0].messages) == 1


def test_mock_error_before_any_output():
    provider = MockProvider([MockTurn(error=RetryableError("boom", 503))])
    with pytest.raises(RetryableError):
        run(provider, request())


def test_mock_error_mid_stream_keeps_events_before_the_failure():
    provider = MockProvider([MockTurn(text="abcdefghij", error=RetryableError("drop"), error_after_events=2)], chunk_size=2)
    seen = []
    cancel = CancelToken()
    with pytest.raises(RetryableError):
        collect(provider.stream(request(), cancel), cancel, on_event=seen.append)
    assert [e.text for e in seen] == ["ab", "cd"]


def test_cancel_mid_stream_reports_partial_message():
    provider = MockProvider(
        [
            MockTurn(
                text="partial answer",
                tool_calls=[MockToolCall("write_file", {"path": "a", "content": "long long long content"})],
            )
        ],
        chunk_size=5,
    )
    cancel = CancelToken()
    seen = []

    def on_event(event):
        seen.append(event)
        # 3 text chunks, then ToolCallStart, then the first argument fragment: mid tool call.
        if len(seen) == 5:
            cancel.cancel()

    with pytest.raises(Cancelled) as info:
        collect(provider.stream(request(), cancel), cancel, on_event=on_event)
    assert info.value.partial is not None
    assert info.value.partial.text == "partial answer"
    assert info.value.partial.tool_uses == []


def test_provider_blocked_waiting_is_released_by_cancel_with_partial_text():
    provider = MockProvider([MockTurn(text="abcdefgh", block_after_events=2)], chunk_size=4)
    cancel = CancelToken()
    timer = threading.Timer(0.05, cancel.cancel)
    timer.start()
    with pytest.raises(Cancelled) as info:
        collect(provider.stream(request(), cancel), cancel)
    timer.join()
    assert info.value.partial is not None and info.value.partial.text == "abcdefgh"


# ---------- error classification ----------


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529, 408])
def test_server_errors_and_timeouts_are_retryable(status):
    assert isinstance(from_http_status(status, "x"), RetryableError)


def test_rate_limit_is_retryable_and_keeps_retry_after():
    error = from_http_status(429, "slow down", retry_after=7)
    assert isinstance(error, RetryableError) and error.retry_after == 7


def test_429_with_exhausted_quota_is_fatal():
    assert isinstance(from_http_status(429, "You exceeded your current quota", code="insufficient_quota"), FatalError)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_fatal_with_clear_message(status):
    error = from_http_status(status, "details")
    assert isinstance(error, FatalError) and not isinstance(error, RetryableError)
    assert str(status) in error.message


def test_401_message_points_at_the_api_key():
    assert "API key" in from_http_status(401, "bad").message


def test_400_context_overflow_is_its_own_error_type():
    error = from_http_status(400, "This model's maximum context length is 8192 tokens", code="context_length_exceeded")
    assert isinstance(error, ContextLengthExceeded) and isinstance(error, FatalError)


def test_parse_retry_after():
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after("-3") is None
