"""Qwen provider: the OpenAI Chat Completions compatible protocol (DashScope compatible mode).

Conversion rules between the internal format and the wire format:
  * assistant tool_use blocks  <->  `tool_calls` on the assistant message (arguments as a JSON string)
  * user tool_result blocks    <->  one `role: "tool"` message per result, placed before any user text
Streaming tool calls arrive in fragments keyed by `index`; id and name come with the first fragment only.
The SDK's own retries are disabled (max_retries=0): retrying is done by providers/retry.py.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

import openai
from pydantic import SecretStr

try:  # the openai SDK ships its own httpx fork
    import httpx2 as _httpx
except ImportError:  # pragma: no cover
    import httpx as _httpx

from mini_agent.agent.cancel import CancelToken
from mini_agent.messages import Message, StopReason, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mini_agent.providers.base import (
    Done,
    ProviderRequest,
    StreamEvent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolSpec,
    UsageEvent,
)
from mini_agent.providers.errors import (
    FatalError,
    ProviderError,
    RetryableError,
    from_http_status,
    parse_retry_after,
)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ------------------------------------------------------------------ request conversion


def _assistant_to_wire(message: Message) -> dict[str, Any]:
    text = message.text
    wire: dict[str, Any] = {"role": "assistant", "content": text if text else None}
    calls = [
        {
            "id": use.id,
            "type": "function",
            "function": {"name": use.name, "arguments": json.dumps(use.input, ensure_ascii=False)},
        }
        for use in message.tool_uses
    ]
    if calls:
        wire["tool_calls"] = calls
    elif wire["content"] is None:
        wire["content"] = ""
    return wire


def _user_to_wire(message: Message) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            content = f"Error: {block.content}" if block.is_error else block.content
            wire.append({"role": "tool", "tool_call_id": block.tool_use_id, "content": content})
    text = message.text
    if text:
        wire.append({"role": "user", "content": text})
    return wire


def to_wire_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    if system:
        wire.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "assistant":
            wire.append(_assistant_to_wire(message))
        else:
            wire.extend(_user_to_wire(message))
    return wire


def to_wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


# ------------------------------------------------------------------ stream conversion


class _Call:
    def __init__(self, index: int, call_id: str) -> None:
        self.index = index
        self.id = call_id
        self.name = ""
        self.started = False
        self.ended = False
        self.pending_args: list[str] = []


def map_finish_reason(finish: str | None, has_calls: bool) -> StopReason:
    if finish == "length":
        return "max_tokens"
    if finish == "content_filter":
        return "error"
    # Some compatible servers answer "stop" even when they return tool calls.
    return "tool_use" if has_calls else "end_turn"


def convert_stream(chunks: Iterable[Any]) -> Iterator[StreamEvent]:
    """Turn SDK `ChatCompletionChunk`s into vendor-neutral StreamEvents."""
    calls: dict[int, _Call] = {}
    order: list[_Call] = []
    used_ids: set[str] = set()
    finish: str | None = None

    def start(call: _Call) -> Iterator[StreamEvent]:
        if not call.started:
            call.started = True
            yield ToolCallStart(call.id, call.name)
            for fragment in call.pending_args:
                yield ToolCallArgsDelta(call.id, fragment)
            call.pending_args.clear()

    for chunk in chunks:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield UsageEvent(Usage(input_tokens=usage.prompt_tokens or 0, output_tokens=usage.completion_tokens or 0))
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if delta is not None:
            # `reasoning_content` (Qwen thinking) is deliberately ignored and never sent back.
            if delta.content:
                yield TextDelta(delta.content)
            for fragment in delta.tool_calls or []:
                index = fragment.index
                if index is None:
                    index = order[-1].index if (order and not fragment.id) else len(order)
                call = calls.get(index)
                if call is None:
                    call_id = fragment.id or ""
                    if not call_id or call_id in used_ids:
                        call_id = f"call_{uuid.uuid4().hex[:12]}"
                    used_ids.add(call_id)
                    call = calls[index] = _Call(index, call_id)
                    order.append(call)
                function = fragment.function
                if function is not None:
                    if function.name and not call.name:
                        call.name = function.name
                    if call.name and not call.started:
                        yield from start(call)
                    if function.arguments:
                        if call.started:
                            yield ToolCallArgsDelta(call.id, function.arguments)
                        else:
                            call.pending_args.append(function.arguments)

        if choice.finish_reason:
            finish = choice.finish_reason
            for call in order:
                if not call.ended:
                    yield from start(call)
                    call.ended = True
                    yield ToolCallEnd(call.id)

    if finish is None:
        raise RetryableError("The response stream ended before the model finished (connection dropped?)")
    yield Done(map_finish_reason(finish, has_calls=bool(order)))


# ------------------------------------------------------------------------- errors


def _error_details(exc: openai.APIStatusError) -> tuple[str, str | None]:
    body = exc.body
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        message = inner.get("message") or exc.message
        code = inner.get("code") or inner.get("type")
        return str(message), (str(code) if code else None)
    return exc.message, None


def _retry_after(exc: openai.APIStatusError) -> float | None:
    headers = exc.response.headers
    seconds = parse_retry_after(headers.get("retry-after"))
    if seconds is None:
        millis = parse_retry_after(headers.get("retry-after-ms"))
        seconds = millis / 1000 if millis is not None else None
    return seconds


def translate_error(exc: BaseException) -> BaseException:
    """Map SDK and transport exceptions to ProviderError; unknown exceptions are returned unchanged."""
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, openai.APIStatusError):
        message, code = _error_details(exc)
        return from_http_status(exc.status_code, message, code=code, retry_after=_retry_after(exc))
    if isinstance(exc, openai.APITimeoutError):
        return RetryableError("Request timed out.")
    if isinstance(exc, openai.APIConnectionError):
        return RetryableError(f"Connection error: {exc.message}")
    if isinstance(exc, openai.APIError):
        # An error event in the middle of a stream.
        probe = f"{exc.code or ''} {exc.message}".lower()
        if any(marker in probe for marker in ("quota", "billing", "arrearage", "invalid", "auth")):
            return FatalError(f"Request failed: {exc.message}")
        return RetryableError(f"Stream error from the server: {exc.message}")
    if isinstance(exc, (_httpx.TransportError, TimeoutError, ConnectionError)):
        return RetryableError(f"Network error: {type(exc).__name__}: {exc}")
    return exc


# ---------------------------------------------------------------------- provider


class QwenProvider:
    name = "qwen"

    def __init__(
        self,
        api_key: str | SecretStr,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        client: openai.OpenAI | None = None,
        read_timeout: float = 120.0,
    ) -> None:
        self.model = model
        if client is None:
            key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
            client = openai.OpenAI(
                api_key=key,
                base_url=base_url,
                max_retries=0,
                # read is the idle time between two chunks of a streaming response
                timeout=openai.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0),
            )
        self._client = client

    def build_request(self, request: ProviderRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_wire_messages(request.system, request.messages),
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            body["tools"] = to_wire_tools(request.tools)
        return body

    def stream(self, request: ProviderRequest, cancel: CancelToken) -> Iterator[StreamEvent]:
        body = self.build_request(request)
        try:
            response = self._client.chat.completions.create(**body)
        except Exception as exc:
            translated = translate_error(exc)
            if translated is exc:
                raise
            raise translated from exc
        try:
            yield from convert_stream(self._chunks(response))
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _chunks(response: Iterable[Any]) -> Iterator[Any]:
        try:
            yield from response
        except Exception as exc:
            translated = translate_error(exc)
            if translated is exc:
                raise
            raise translated from exc
