import base64
import json

import httpx
import pytest
from google import genai
from google.genai import types
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
from mini_agent.providers.gemini import (
    GeminiProvider,
    convert_stream,
    to_gemini_contents,
    to_gemini_tools,
)
from mini_agent.providers.retry import RetryPolicy, call_with_retry
from mini_agent.tools.registry import default_registry
from mini_agent.tools.sandbox import Sandbox

API_KEY = "AIzaSyTestKey0123456789abcdefghijk"
SIG = b"opaque-signature-bytes\x00\xff"
SIG_B64 = base64.b64encode(SIG).decode()


# ------------------------------------------------------------------ helpers


def part_text(text):
    return types.Part(text=text)


def part_call(name, args, call_id=None, signature=None):
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args), thought_signature=signature)


def response(parts=None, finish=None, usage=None, finish_message=None):
    candidates = None
    if parts is not None or finish is not None:
        candidates = [
            types.Candidate(
                content=types.Content(role="model", parts=parts) if parts is not None else None,
                finish_reason=finish,
                finish_message=finish_message,
            )
        ]
    metadata = None
    if usage:
        metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage[0], candidates_token_count=usage[1], thoughts_token_count=usage[2] if len(usage) > 2 else None
        )
    return types.GenerateContentResponse(candidates=candidates, usage_metadata=metadata)


def run_chunks(*chunks):
    return collect(iter(convert_stream(chunks)), CancelToken())


def wire_call(name, args, signature=None):
    part = {"functionCall": {"name": name, "args": args}}
    if signature:
        part["thoughtSignature"] = signature
    return part


def wire_chunk(parts=None, finish=None, usage=None):
    candidate = {"content": {"role": "model", "parts": parts or []}}
    if finish:
        candidate["finishReason"] = finish
    out = {"candidates": [candidate]}
    if usage:
        out["usageMetadata"] = {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1], "thoughtsTokenCount": usage[2]}
    return out


def sse_response(*events):
    body = "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events).encode()
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def make_provider(handler):
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    return GeminiProvider(API_KEY, "gemini-test", client=client)


def request(*messages, tools=None, system="sys"):
    return ProviderRequest(system=system, messages=list(messages) or [Message.user("hi")], tools=tools or [])


def run_provider(provider, req=None):
    cancel = CancelToken()
    return collect(provider.stream(req or request(), cancel), cancel)


def error_response(status, message, status_name, details=None, headers=None):
    error = {"code": status, "message": message, "status": status_name}
    if details:
        error["details"] = details
    return lambda req: httpx.Response(status, json={"error": error}, headers=headers or {})


# ------------------------------------------------- request conversion (pure)


def history_with_calls():
    return [
        Message.user("fix it"),
        Message(
            role="assistant",
            content=[
                TextBlock(text="checking"),
                ToolUseBlock(id="call_a", name="read_file", input={"path": "你好.py"}, meta={"thought_signature": SIG_B64, "id_generated": True}),
                ToolUseBlock(id="real_b", name="grep", input={"pattern": "x"}),
            ],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="call_a", content="file data"),
                ToolResultBlock(tool_use_id="real_b", content="no match", is_error=True),
                TextBlock(text="and then?"),
            ],
        ),
    ]


def test_roles_are_user_and_model_and_text_stays_text():
    contents = to_gemini_contents([Message.user("hi"), Message.assistant("hello")])
    assert [c.role for c in contents] == ["user", "model"]
    assert contents[0].parts[0].text == "hi" and contents[1].parts[0].text == "hello"


def test_tool_use_becomes_function_call_parts_with_signature_restored_to_bytes():
    model = to_gemini_contents(history_with_calls())[1]
    assert [bool(p.text) for p in model.parts] == [True, False, False]
    first, second = model.parts[1], model.parts[2]
    assert first.function_call.name == "read_file" and first.function_call.args == {"path": "你好.py"}
    assert first.thought_signature == SIG
    assert second.thought_signature is None  # only the first parallel call carries a signature


def test_generated_ids_are_not_sent_but_real_ids_are():
    contents = to_gemini_contents(history_with_calls())
    model, user = contents[1], contents[2]
    assert model.parts[1].function_call.id is None and model.parts[2].function_call.id == "real_b"
    responses = [p.function_response for p in user.parts if p.function_response]
    assert [r.id for r in responses] == [None, "real_b"]


def test_function_responses_use_names_from_earlier_tool_calls_and_come_before_text():
    user = to_gemini_contents(history_with_calls())[2]
    assert [bool(p.function_response) for p in user.parts] == [True, True, False]
    first, second = user.parts[0].function_response, user.parts[1].function_response
    assert (first.name, first.response) == ("read_file", {"output": "file data"})
    assert (second.name, second.response) == ("grep", {"error": "no match"})
    assert user.parts[2].text == "and then?"


def test_function_response_names_are_found_across_several_turns():
    history = [
        Message.user("a"),
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="list_files", input={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="x")]),
        Message.assistant("done"),
        Message.user("b"),
        Message(role="assistant", content=[ToolUseBlock(id="c2", name="grep", input={"pattern": "y"})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="c2", content="z")]),
    ]
    contents = to_gemini_contents(history)
    assert contents[2].parts[0].function_response.name == "list_files"
    assert contents[6].parts[0].function_response.name == "grep"


def test_tool_use_with_invalid_arguments_is_sent_with_empty_args():
    broken = ToolUseBlock.from_raw("c1", "write_file", '{"path": "x", "content": "abc')
    model = to_gemini_contents([Message.user("x"), Message(role="assistant", content=[broken])])[1]
    assert model.parts[0].function_call.args == {}


def test_tool_specs_become_classic_schema_objects():
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File."},
            "n": {"type": "integer", "minimum": 1, "maximum": 9},
            "flag": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["path"],
    }
    tool = to_gemini_tools([ToolSpec(name="read_file", description="Read.", parameters=schema)])[0]
    declaration = tool.function_declarations[0]
    assert declaration.name == "read_file" and declaration.description == "Read."
    assert declaration.parameters_json_schema is None  # the newer field is deliberately not used
    parameters = declaration.parameters
    assert parameters.type == types.Type.OBJECT and parameters.required == ["path"]
    assert parameters.properties["path"].type == types.Type.STRING and parameters.properties["path"].description == "File."
    assert (parameters.properties["n"].type, parameters.properties["n"].minimum, parameters.properties["n"].maximum) == (types.Type.INTEGER, 1, 9)
    assert parameters.properties["flag"].type == types.Type.BOOLEAN
    assert parameters.properties["mode"].enum == ["a", "b"]
    assert to_gemini_tools([]) == []


def test_every_real_tool_schema_converts_without_losing_properties_or_required_fields():
    for spec in default_registry().specs():
        converted = to_gemini_tools([spec])[0].function_declarations[0].parameters
        assert set(converted.properties) == set(spec.parameters["properties"]), spec.name
        assert (converted.required or []) == spec.parameters.get("required", []), spec.name
        assert converted.type == types.Type.OBJECT


# ------------------------------------------------- stream conversion (pure)


def test_text_is_streamed_and_thought_summaries_are_dropped():
    result = run_chunks(
        response([part_text("Hel")]),
        response([types.Part(text="private reasoning", thought=True), part_text("lo")]),
        response([part_text("!")], finish="STOP"),
    )
    assert result.message.text == "Hello!" and result.stop_reason == "end_turn"
    assert "private reasoning" not in result.message.model_dump_json()


def test_function_call_becomes_a_tool_use_with_generated_id_and_signature_in_meta():
    result = run_chunks(
        response([part_text("Looking. "), part_call("read_file", {"path": "a.py"}, signature=SIG)], finish="STOP")
    )
    use = result.message.tool_uses[0]
    assert (use.name, use.input) == ("read_file", {"path": "a.py"})
    assert use.id.startswith("call_") and use.meta == {"id_generated": True, "thought_signature": SIG_B64}
    assert result.stop_reason == "tool_use"


def test_real_function_call_ids_are_kept_and_duplicates_are_replaced():
    result = run_chunks(
        response([part_call("grep", {}, "x1"), part_call("grep", {}, "x1"), part_call("grep", {})], finish="STOP")
    )
    ids = [u.id for u in result.message.tool_uses]
    assert ids[0] == "x1" and len(set(ids)) == 3
    assert result.message.tool_uses[0].meta == {} and result.message.tool_uses[1].meta["id_generated"] is True


def test_parallel_calls_arrive_in_separate_chunks_and_keep_their_order():
    result = run_chunks(
        response([part_call("read_file", {"path": "a"}, signature=SIG)]),
        response([part_call("grep", {"pattern": "b"})], finish="STOP"),
    )
    assert [(u.name, u.input) for u in result.message.tool_uses] == [("read_file", {"path": "a"}), ("grep", {"pattern": "b"})]


def test_usage_adds_thinking_tokens_to_the_output_and_the_last_chunk_wins():
    result = run_chunks(
        response([part_text("a")], usage=(10, 1, 0)),
        response([part_text("b")], finish="STOP", usage=(10, 4, 30)),
    )
    assert result.usage == Usage(input_tokens=10, output_tokens=34)


def test_missing_usage_stays_none_so_the_agent_can_estimate():
    assert run_chunks(response([part_text("a")], finish="STOP")).usage is None


def test_max_tokens_with_partial_text_is_reported_as_max_tokens():
    result = run_chunks(response([part_text("cut o")], finish="MAX_TOKENS"))
    assert result.stop_reason == "max_tokens" and result.message.text == "cut o"


def test_max_tokens_before_any_visible_output_explains_the_thinking_budget():
    with pytest.raises(FatalError, match="max_tokens"):
        run_chunks(response([], finish="MAX_TOKENS"))


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "OTHER", "BLOCKLIST"])
def test_blocked_or_unusual_finish_reasons_are_fatal_with_the_reason_named(finish):
    with pytest.raises(FatalError) as info:
        run_chunks(response([part_text("x")], finish=finish, finish_message="details here"))
    assert finish in info.value.message and "details here" in info.value.message
    assert not isinstance(info.value, RetryableError)


def test_malformed_function_call_is_retryable_and_missing_signature_is_fatal():
    with pytest.raises(RetryableError):
        run_chunks(response([], finish="MALFORMED_FUNCTION_CALL"))
    with pytest.raises(FatalError, match="thought signature"):
        run_chunks(response([], finish="MISSING_THOUGHT_SIGNATURE"))


def test_stream_without_finish_reason_is_a_retryable_drop():
    with pytest.raises(RetryableError):
        run_chunks(response([part_text("partial")]))


def test_blocked_prompt_is_fatal():
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY", block_reason_message="unsafe prompt")
    )
    with pytest.raises(FatalError, match="blocked the prompt"):
        run_chunks(blocked)


# ------------------------- real SDK + mock HTTP transport (full wire path)


def test_request_body_and_headers_on_the_wire():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["key"] = req.headers["x-goog-api-key"]
        seen["body"] = json.loads(req.content)
        return sse_response(wire_chunk([{"text": "ok"}], finish="STOP"))

    spec_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    spec = ToolSpec(name="read_file", description="Read.", parameters=spec_schema)
    run_provider(make_provider(handler), request(tools=[spec]))
    body = seen["body"]
    assert ":streamGenerateContent" in seen["url"] and "alt=sse" in seen["url"] and "gemini-test" in seen["url"]
    assert seen["key"] == API_KEY and API_KEY not in json.dumps(body) and API_KEY not in seen["url"]
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["generationConfig"]["maxOutputTokens"] == 4096
    # the whole declaration: the parameter schema must really reach the model
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "read_file",
                    "description": "Read.",
                    "parameters": {"type": "OBJECT", "properties": {"path": {"type": "STRING"}}, "required": ["path"]},
                }
            ]
        }
    ]


def test_sse_stream_is_parsed_by_the_real_sdk_into_a_complete_message():
    def handler(req):
        return sse_response(
            wire_chunk([{"text": "Looking now. "}]),
            wire_chunk([wire_call("read_file", {"path": "a.py"}, SIG_B64)], finish="STOP", usage=(20, 5, 30)),
        )

    result = run_provider(make_provider(handler))
    assert result.message.text == "Looking now. "
    use = result.message.tool_uses[0]
    assert use.input == {"path": "a.py"} and use.meta["thought_signature"] == SIG_B64
    assert result.stop_reason == "tool_use" and result.usage == Usage(input_tokens=20, output_tokens=35)


def test_429_per_minute_is_retryable_with_the_servers_retry_delay():
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]
    handler = error_response(429, "You exceeded your current quota, please check your plan and billing details.", "RESOURCE_EXHAUSTED", details)
    with pytest.raises(RetryableError) as info:
        run_provider(make_provider(handler))
    assert info.value.status_code == 429 and info.value.retry_after == 34


def test_429_daily_quota_exhausted_is_fatal_because_waiting_does_not_help():
    details = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
    ]
    with pytest.raises(FatalError) as info:
        run_provider(make_provider(error_response(429, "quota", "RESOURCE_EXHAUSTED", details)))
    assert "daily quota" in info.value.message and not isinstance(info.value, RetryableError)


def test_an_invalid_api_key_is_reported_as_an_authentication_problem():
    handler = error_response(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    with pytest.raises(FatalError) as info:
        run_provider(make_provider(handler))
    assert "Authentication failed" in info.value.message and "API key" in info.value.message
    assert not isinstance(info.value, RetryableError) and API_KEY not in info.value.message


def test_other_client_errors_are_fatal_and_context_overflow_is_recognised():
    with pytest.raises(FatalError) as info:
        run_provider(make_provider(error_response(404, "models/nope is not found", "NOT_FOUND")))
    assert info.value.status_code == 404 and "model name" in info.value.message

    with pytest.raises(FatalError) as info:
        run_provider(make_provider(error_response(400, "Request contains an invalid argument.", "INVALID_ARGUMENT")))
    assert info.value.status_code == 400 and not isinstance(info.value, ContextLengthExceeded)

    overflow = error_response(400, "The input token count (2000000) exceeds the maximum number of tokens allowed (1048576).", "INVALID_ARGUMENT")
    with pytest.raises(ContextLengthExceeded):
        run_provider(make_provider(overflow))


@pytest.mark.parametrize("status, name", [(500, "INTERNAL"), (503, "UNAVAILABLE"), (504, "DEADLINE_EXCEEDED")])
def test_5xx_is_retryable(status, name):
    with pytest.raises(RetryableError) as info:
        run_provider(make_provider(error_response(status, "try later", name)))
    assert info.value.status_code == status


def test_network_failures_and_timeouts_are_retryable():
    def refused(req):
        raise httpx.ConnectError("refused", request=req)

    def too_slow(req):
        raise httpx.ReadTimeout("slow", request=req)

    for handler in (refused, too_slow):
        with pytest.raises(RetryableError):
            run_provider(make_provider(handler))


def test_connection_dropped_in_the_middle_of_the_stream_is_retryable():
    def body():
        yield b"data: " + json.dumps(wire_chunk([{"text": "par"}])).encode() + b"\r\n\r\n"
        raise httpx.RemoteProtocolError("peer closed connection")

    provider = make_provider(lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))
    with pytest.raises(RetryableError):
        run_provider(provider)


def test_error_event_inside_the_stream_is_classified_like_an_http_error():
    def handler(req):
        event = {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(event)}\r\n\r\n".encode())

    with pytest.raises(RetryableError) as info:
        run_provider(make_provider(handler))
    assert info.value.status_code == 503


def test_rate_limit_then_success_through_the_retry_layer_uses_the_servers_delay():
    calls = []
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5s"}]

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return error_response(429, "slow down", "RESOURCE_EXHAUSTED", details)(req)
        return sse_response(wire_chunk([{"text": "finally"}], finish="STOP"))

    provider = make_provider(handler)
    cancel = CancelToken()
    sleeps = []
    result = call_with_retry(lambda: collect(provider.stream(request(), cancel), cancel), RetryPolicy(jitter=0), cancel, sleep=sleeps.append)
    assert result.message.text == "finally" and len(calls) == 3 and sleeps == [5.0, 5.0]


def test_sdk_level_retries_are_disabled_so_each_failure_is_one_http_request():
    calls = []

    def handler(req):
        calls.append(1)
        return error_response(503, "down", "UNAVAILABLE")(req)

    provider = make_provider(handler)
    cancel = CancelToken()
    with pytest.raises(RetriesExhausted):
        call_with_retry(lambda: collect(provider.stream(request(), cancel), cancel), RetryPolicy(max_retries=2, jitter=0), cancel, sleep=lambda d: None)
    assert len(calls) == 3  # exactly max_retries + 1: the SDK added no hidden retries


def test_the_default_client_is_built_with_sdk_retries_off_and_no_key_in_repr():
    provider = GeminiProvider(API_KEY, "gemini-test")
    options = provider._client._api_client._http_options
    assert options.retry_options.attempts == 1 and options.timeout == 120_000
    assert API_KEY not in repr(provider.__dict__.get("model")) + provider.name


def test_unexpected_exceptions_are_not_disguised_as_network_errors():
    def bug(req):
        raise ValueError("test bug")

    with pytest.raises(ValueError, match="test bug"):
        run_provider(make_provider(bug))


# --------------------------- agent loop on top of the real SDK wire path


def test_agent_round_trip_echoes_the_thought_signature_and_names_the_function_response(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text("hello from file\n")
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        if len(bodies) == 1:
            return sse_response(
                wire_chunk([{"text": "Let me read it. "}, wire_call("read_file", {"path": "a.txt"}, SIG_B64)], finish="STOP", usage=(7, 3, 10))
            )
        return sse_response(wire_chunk([{"text": "It says hello."}], finish="STOP", usage=(12, 4, 0)))

    ui = RecordingUI()
    agent = Agent(make_provider(handler), default_registry(), Sandbox(work), ui, build_system_prompt(str(work)), AgentConfig())
    result = agent.run_turn("what is in a.txt?")
    assert result.status == "completed" and result.text == "It says hello."
    assert result.usage == Usage(input_tokens=19, output_tokens=17)

    second = bodies[1]["contents"]
    assert [c["role"] for c in second] == ["user", "model", "user"]
    model_parts = second[1]["parts"]
    assert model_parts[0] == {"text": "Let me read it. "}
    assert model_parts[1]["functionCall"] == {"name": "read_file", "args": {"path": "a.txt"}}  # generated id is not sent
    # The SDK writes bytes as URL-safe base64; what matters is that the decoded bytes are identical.
    assert base64.urlsafe_b64decode(model_parts[1]["thoughtSignature"]) == SIG
    function_response = second[2]["parts"][0]["functionResponse"]
    assert function_response["name"] == "read_file" and "hello from file" in function_response["response"]["output"]
    assert "id" not in function_response


def test_agent_reports_an_invalid_key_clearly_through_the_whole_stack(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    provider = make_provider(error_response(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"))
    ui = RecordingUI()
    agent = Agent(provider, default_registry(), Sandbox(work), ui, "sys", AgentConfig(), sleep=lambda d: None)
    result = agent.run_turn("hello")
    assert result.status == "error" and "Authentication failed" in result.error
    assert ui.retries == []
