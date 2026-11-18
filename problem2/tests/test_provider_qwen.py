import json

import httpx2
import openai
import pytest
from helpers import RecordingUI

from mini_agent.agent.cancel import CancelToken
from mini_agent.agent.loop import Agent
from mini_agent.agent.prompt import build_system_prompt
from mini_agent.config import AgentConfig
from mini_agent.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mini_agent.providers.base import ProviderRequest, ToolSpec, collect
from mini_agent.providers.errors import (
    ContextLengthExceeded,
    FatalError,
    RetriesExhausted,
    RetryableError,
)
from mini_agent.providers.qwen import (
    QwenProvider,
    convert_stream,
    map_finish_reason,
    to_wire_messages,
    to_wire_tools,
)
from mini_agent.providers.retry import RetryPolicy, call_with_retry
from mini_agent.tools.registry import default_registry
from mini_agent.tools.sandbox import Sandbox

API_KEY = "sk-test-qwen-0123456789abcdef"


# ------------------------------------------------------------------ helpers


def chunk(delta=None, finish=None, usage=None, with_choice=True):
    data = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen-plus",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if with_choice else [],
    }
    if usage:
        data["usage"] = usage
    return data


def tc(index, call_id=None, name=None, args=None):
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    out = {"index": index, "function": fn}
    if call_id is not None:
        out["id"] = call_id
        out["type"] = "function"
    return out


def sdk_chunks(*raw):
    from openai.types.chat import ChatCompletionChunk

    return [ChatCompletionChunk.model_validate(r) for r in raw]


def sse(*events) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def stream_events(*raw):
    return list(convert_stream(sdk_chunks(*raw)))


def run_chunks(*raw):
    cancel = CancelToken()
    return collect(iter(convert_stream(sdk_chunks(*raw))), cancel)


def make_provider(handler, **kwargs):
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="http://qwen.test/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    return QwenProvider(API_KEY, "qwen-plus", client=client, **kwargs)


def sse_response(*events):
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse(*events))


def request(*messages, tools=None):
    return ProviderRequest(system="sys", messages=list(messages) or [Message.user("hi")], tools=tools or [])


def run_provider(provider, req=None):
    cancel = CancelToken()
    return collect(provider.stream(req or request(), cancel), cancel)


# ------------------------------------------------- request conversion (pure)


def test_system_prompt_comes_first_and_empty_system_is_omitted():
    assert to_wire_messages("be brief", [Message.user("hi")]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert to_wire_messages("", [Message.user("hi")]) == [{"role": "user", "content": "hi"}]


def test_tool_use_becomes_tool_calls_and_results_become_tool_messages_in_order():
    history = [
        Message.user("go"),
        Message(
            role="assistant",
            content=[
                TextBlock(text="checking"),
                ToolUseBlock(id="a", name="read_file", input={"path": "你好.txt"}),
                ToolUseBlock(id="b", name="grep", input={"pattern": "x"}),
            ],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="a", content="file data"),
                ToolResultBlock(tool_use_id="b", content="nothing", is_error=True),
            ],
        ),
    ]
    wire = to_wire_messages("", history)
    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "tool"]
    assistant = wire[1]
    assert assistant["content"] == "checking"
    assert [c["id"] for c in assistant["tool_calls"]] == ["a", "b"]
    assert assistant["tool_calls"][0]["type"] == "function"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"path": "你好.txt"}
    assert "你好" in assistant["tool_calls"][0]["function"]["arguments"]  # not \u-escaped
    assert wire[2] == {"role": "tool", "tool_call_id": "a", "content": "file data"}
    assert wire[3] == {"role": "tool", "tool_call_id": "b", "content": "Error: nothing"}


def test_assistant_with_only_tool_calls_has_null_content_and_text_only_has_no_tool_calls():
    only_calls = Message(role="assistant", content=[ToolUseBlock(id="a", name="list_files", input={})])
    wire = to_wire_messages("", [Message.user("x"), only_calls])
    assert wire[1]["content"] is None and wire[1]["tool_calls"][0]["function"]["arguments"] == "{}"
    text_only = to_wire_messages("", [Message.user("x"), Message.assistant("hello")])
    assert text_only[1] == {"role": "assistant", "content": "hello"}


def test_tool_use_with_unparseable_arguments_is_sent_as_an_empty_object():
    broken = ToolUseBlock.from_raw("a", "write_file", '{"path": "x", "content": "abc')
    wire = to_wire_messages("", [Message.user("x"), Message(role="assistant", content=[broken])])
    assert wire[1]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_tool_results_come_before_user_text_in_the_same_message():
    message = Message(role="user", content=[ToolResultBlock(tool_use_id="a", content="r"), TextBlock(text="note")])
    wire = to_wire_messages("", [Message.user("x"), Message(role="assistant", content=[ToolUseBlock(id="a", name="n", input={})]), message])
    assert [m["role"] for m in wire[2:]] == ["tool", "user"]


def test_tool_specs_use_the_function_calling_format():
    spec = ToolSpec(name="read_file", description="Read.", parameters={"type": "object", "properties": {}})
    assert to_wire_tools([spec]) == [
        {"type": "function", "function": {"name": "read_file", "description": "Read.", "parameters": {"type": "object", "properties": {}}}}
    ]


# ------------------------------------------------- stream conversion (pure)


def test_text_deltas_are_streamed_and_reasoning_content_is_ignored():
    result = run_chunks(
        chunk({"role": "assistant", "content": "Hel", "reasoning_content": "secret thoughts"}),
        chunk({"content": "lo"}),
        chunk({"reasoning_content": "more thoughts"}),
        chunk({}, finish="stop"),
    )
    assert result.message.text == "Hello" and result.stop_reason == "end_turn"
    assert "thoughts" not in result.message.model_dump_json()


def test_tool_call_arguments_are_joined_across_fragments():
    result = run_chunks(
        chunk({"tool_calls": [tc(0, "call_1", "read_file", "")]}),
        chunk({"tool_calls": [tc(0, args='{"pa')]}),
        chunk({"tool_calls": [tc(0, args='th": "a.')]}),
        chunk({"tool_calls": [tc(0, args='py"}')]}),
        chunk({}, finish="tool_calls"),
    )
    use = result.message.tool_uses[0]
    assert (use.id, use.name, use.input) == ("call_1", "read_file", {"path": "a.py"})
    assert result.stop_reason == "tool_use"


def test_parallel_tool_calls_with_interleaved_fragments_stay_separate():
    result = run_chunks(
        chunk({"tool_calls": [tc(0, "c0", "read_file", ""), tc(1, "c1", "grep", "")]}),
        chunk({"tool_calls": [tc(1, args='{"pattern":')]}),
        chunk({"tool_calls": [tc(0, args='{"path": "a"}')]}),
        chunk({"tool_calls": [tc(1, args=' "x"}')]}),
        chunk({}, finish="tool_calls"),
    )
    assert [(u.id, u.name, u.input) for u in result.message.tool_uses] == [
        ("c0", "read_file", {"path": "a"}),
        ("c1", "grep", {"pattern": "x"}),
    ]


def test_text_followed_by_tool_call_keeps_both_in_order():
    result = run_chunks(
        chunk({"content": "I will read it."}),
        chunk({"tool_calls": [tc(0, "c0", "read_file", '{"path": "a"}')]}),
        chunk({}, finish="tool_calls"),
    )
    assert [b.type for b in result.message.content] == ["text", "tool_use"]


def test_finish_reason_length_marks_truncated_arguments_as_invalid():
    result = run_chunks(
        chunk({"tool_calls": [tc(0, "c0", "write_file", '{"path": "big.txt", "content": "abc')]}),
        chunk({}, finish="length"),
    )
    use = result.message.tool_uses[0]
    assert result.stop_reason == "max_tokens"
    assert use.input == {} and use.raw_arguments.startswith('{"path"') and not use.arguments_valid


def test_usage_arrives_in_a_final_chunk_without_choices():
    result = run_chunks(
        chunk({"content": "hi"}),
        chunk({}, finish="stop"),
        chunk(usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}, with_choice=False),
    )
    assert result.usage == Usage(input_tokens=11, output_tokens=7)


def test_missing_usage_stays_none_so_the_agent_can_estimate():
    assert run_chunks(chunk({"content": "hi"}), chunk({}, finish="stop")).usage is None


def test_server_that_says_stop_but_returned_tool_calls_is_treated_as_tool_use():
    result = run_chunks(chunk({"tool_calls": [tc(0, "c0", "list_files", "{}")]}), chunk({}, finish="stop"))
    assert result.stop_reason == "tool_use"


def test_arguments_before_the_name_are_buffered_until_the_name_arrives():
    result = run_chunks(
        chunk({"tool_calls": [tc(0, "c0", None, '{"path": ')]}),
        chunk({"tool_calls": [tc(0, None, "read_file", '"a"}')]}),
        chunk({}, finish="tool_calls"),
    )
    assert result.message.tool_uses[0].name == "read_file" and result.message.tool_uses[0].input == {"path": "a"}


def test_missing_or_duplicate_call_ids_are_replaced_with_unique_ones():
    result = run_chunks(
        chunk({"tool_calls": [tc(0, "dup", "list_files", "{}"), tc(1, "dup", "list_files", "{}"), tc(2, None, "list_files", "{}")]}),
        chunk({}, finish="tool_calls"),
    )
    ids = [u.id for u in result.message.tool_uses]
    assert len(set(ids)) == 3 and ids[0] == "dup" and all(ids)


@pytest.mark.parametrize(
    "finish, has_calls, expected",
    [("stop", False, "end_turn"), ("tool_calls", True, "tool_use"), ("length", True, "max_tokens"),
     ("content_filter", False, "error"), ("weird", False, "end_turn")],
)
def test_finish_reason_mapping(finish, has_calls, expected):
    assert map_finish_reason(finish, has_calls) == expected


def test_stream_without_finish_reason_is_a_retryable_drop():
    with pytest.raises(RetryableError):
        run_chunks(chunk({"content": "partial"}))


# ------------------------- real SDK + mock HTTP transport (full wire path)


def test_request_body_and_headers_on_the_wire():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        seen["url"] = str(req.url)
        return sse_response(chunk({"content": "ok"}), chunk({}, finish="stop"))

    spec = ToolSpec(name="read_file", description="Read.", parameters={"type": "object", "properties": {}})
    run_provider(make_provider(handler), request(tools=[spec]))
    body = seen["body"]
    assert seen["url"] == "http://qwen.test/v1/chat/completions"
    assert seen["auth"] == f"Bearer {API_KEY}"
    assert body["model"] == "qwen-plus" and body["stream"] is True and body["max_tokens"] == 4096
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert API_KEY not in json.dumps(body)


def test_sse_stream_is_parsed_by_the_real_sdk_into_a_complete_message():
    def handler(req):
        return sse_response(
            chunk({"role": "assistant", "content": "Looking"}),
            chunk({"content": " now.", "reasoning_content": "hmm"}),
            chunk({"tool_calls": [tc(0, "call_9", "read_file", "")]}),
            chunk({"tool_calls": [tc(0, args='{"path":')]}),
            chunk({"tool_calls": [tc(0, args=' "a.py"}')]}),
            chunk({}, finish="tool_calls"),
            chunk(usage={"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}, with_choice=False),
        )

    result = run_provider(make_provider(handler))
    assert result.message.text == "Looking now."
    assert result.message.tool_uses[0].input == {"path": "a.py"} and result.message.tool_uses[0].id == "call_9"
    assert result.stop_reason == "tool_use" and result.usage == Usage(input_tokens=20, output_tokens=9)


def error_handler(status, payload, headers=None):
    return lambda req: httpx2.Response(status, headers=headers or {}, json=payload)


def test_401_is_fatal_with_a_clear_message_that_does_not_leak_the_key():
    provider = make_provider(error_handler(401, {"error": {"message": "Incorrect API key provided.", "code": "invalid_api_key"}}))
    with pytest.raises(FatalError) as info:
        run_provider(provider)
    assert info.value.status_code == 401 and "API key" in info.value.message
    assert not isinstance(info.value, RetryableError)
    assert API_KEY not in info.value.message


def test_400_is_fatal_and_context_overflow_is_recognised():
    provider = make_provider(error_handler(400, {"error": {"message": "Bad field", "code": "invalid_parameter_error"}}))
    with pytest.raises(FatalError) as info:
        run_provider(provider)
    assert info.value.status_code == 400 and not isinstance(info.value, ContextLengthExceeded)

    overflow = make_provider(
        error_handler(400, {"error": {"message": "Range of input length should be [1, 98304]", "code": "invalid_parameter_error"}})
    )
    with pytest.raises(ContextLengthExceeded):
        run_provider(overflow)


def test_429_is_retryable_and_keeps_retry_after_but_exhausted_quota_is_fatal():
    limited = make_provider(
        error_handler(429, {"error": {"message": "Requests rate limit exceeded", "code": "limit_burst_rate"}}, {"retry-after": "7"})
    )
    with pytest.raises(RetryableError) as info:
        run_provider(limited)
    assert info.value.status_code == 429 and info.value.retry_after == 7

    broke = make_provider(error_handler(429, {"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}}))
    with pytest.raises(FatalError) as info:
        run_provider(broke)
    assert not isinstance(info.value, RetryableError)


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_is_retryable(status):
    provider = make_provider(error_handler(status, {"error": {"message": "server busy"}}))
    with pytest.raises(RetryableError) as info:
        run_provider(provider)
    assert info.value.status_code == status


def test_network_failures_and_timeouts_are_retryable():
    def refused(req):
        raise httpx2.ConnectError("refused", request=req)

    def too_slow(req):
        raise httpx2.ReadTimeout("slow", request=req)

    for handler in (refused, too_slow):
        with pytest.raises(RetryableError):
            run_provider(make_provider(handler))


def test_connection_dropped_in_the_middle_of_the_stream_is_retryable():
    def body():
        yield b"data: " + json.dumps(chunk({"content": "par"})).encode() + b"\n\n"
        raise httpx2.RemoteProtocolError("peer closed connection")

    provider = make_provider(lambda req: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body()))
    with pytest.raises(RetryableError):
        run_provider(provider)


def test_rate_limit_then_success_through_the_retry_layer():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx2.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down", "code": "limit_burst_rate"}})
        return sse_response(chunk({"content": "finally"}), chunk({}, finish="stop"))

    provider = make_provider(handler)
    cancel = CancelToken()
    sleeps = []
    result = call_with_retry(
        lambda: collect(provider.stream(request(), cancel), cancel), RetryPolicy(jitter=0), cancel, sleep=sleeps.append
    )
    assert result.message.text == "finally" and len(calls) == 3
    assert sleeps == [2.0, 2.0]  # the server's Retry-After wins over the 1s/2s backoff floor


def test_sdk_level_retries_are_disabled_so_each_failure_is_one_http_request():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx2.Response(503, json={"error": {"message": "down"}})

    cancel = CancelToken()
    provider = make_provider(handler)
    with pytest.raises(RetriesExhausted):
        call_with_retry(lambda: collect(provider.stream(request(), cancel), cancel), RetryPolicy(max_retries=2, jitter=0), cancel, sleep=lambda d: None)
    assert len(calls) == 3  # exactly max_retries + 1: the SDK added no hidden retries


def test_unexpected_exceptions_are_not_disguised_as_network_errors():
    def bug(req):
        raise ValueError("test bug")

    # A programming error must stay visible (the agent loop reports it); retrying it would hide the bug.
    with pytest.raises(ValueError, match="test bug"):
        run_provider(make_provider(bug))


# --------------------------- agent loop on top of the real SDK wire path


def test_agent_round_trip_sends_tool_results_back_in_the_openai_format(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text("hello from file\n")
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if len(bodies) == 1:
            return sse_response(
                chunk({"tool_calls": [tc(0, "call_1", "read_file", '{"path": "a.txt"}')]}),
                chunk({}, finish="tool_calls"),
                chunk(usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}, with_choice=False),
            )
        return sse_response(chunk({"content": "It says hello."}), chunk({}, finish="stop"),
                            chunk(usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}, with_choice=False))

    ui = RecordingUI()
    agent = Agent(make_provider(handler), default_registry(), Sandbox(work), ui, build_system_prompt(str(work)), AgentConfig())
    result = agent.run_turn("what is in a.txt?")
    assert result.status == "completed" and result.text == "It says hello."
    assert result.usage == Usage(input_tokens=14, output_tokens=7)

    second = bodies[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["id"] == "call_1"
    assert second[3]["tool_call_id"] == "call_1" and "hello from file" in second[3]["content"]
    assert {t["function"]["name"] for t in bodies[0]["tools"]} == {"list_files", "read_file", "write_file", "edit_file", "grep", "run_shell"}


def test_agent_reports_a_401_clearly_through_the_whole_stack(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    provider = make_provider(error_handler(401, {"error": {"message": "Incorrect API key provided.", "code": "invalid_api_key"}}))
    ui = RecordingUI()
    agent = Agent(provider, default_registry(), Sandbox(work), ui, "sys", AgentConfig(), sleep=lambda d: None)
    result = agent.run_turn("hello")
    assert result.status == "error" and "401" in result.error and "API key" in result.error
    assert ui.retries == []
