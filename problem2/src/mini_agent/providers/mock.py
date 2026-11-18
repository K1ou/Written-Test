"""Scripted provider for tests: no network, no API key.

Each call to `stream` consumes the next MockTurn. Like a real API it rejects an invalid
message history with a 400, so every test that uses it doubles as a history-validity check.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mini_agent.agent.cancel import CancelToken, Cancelled
from mini_agent.history import validate
from mini_agent.messages import StopReason, Usage
from mini_agent.providers.base import (
    Done,
    ProviderRequest,
    StreamEvent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from mini_agent.providers.errors import FatalError


@dataclass
class MockToolCall:
    name: str
    # A dict is serialized to JSON; a str is sent as-is (use it for truncated/invalid JSON).
    arguments: dict[str, Any] | str = field(default_factory=dict)
    id: str | None = None


@dataclass
class MockTurn:
    text: str = ""
    tool_calls: list[MockToolCall] = field(default_factory=list)
    stop_reason: StopReason | None = None
    usage: Usage | None = None
    # Raise `error` after yielding `error_after_events` events (0 = before any output).
    error: Exception | None = None
    error_after_events: int = 0
    # After yielding this many events, block until the turn is cancelled.
    block_after_events: int | None = None


class MockProvider:
    name = "mock"

    def __init__(self, script: list[MockTurn], model: str = "mock-model", chunk_size: int = 7) -> None:
        self.model = model
        self._script = list(script)
        self._chunk_size = chunk_size
        self._call_counter = 0
        self.requests: list[ProviderRequest] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def _chunks(self, text: str) -> list[str]:
        size = self._chunk_size
        return [text[i : i + size] for i in range(0, len(text), size)]

    def _events(self, turn: MockTurn) -> list[StreamEvent]:
        events: list[StreamEvent] = [TextDelta(c) for c in self._chunks(turn.text)]
        for call in turn.tool_calls:
            self._call_counter += 1
            call_id = call.id or f"call_{self._call_counter}"
            raw = call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments)
            events.append(ToolCallStart(call_id, call.name))
            events.extend(ToolCallArgsDelta(call_id, c) for c in self._chunks(raw))
            events.append(ToolCallEnd(call_id))
        events.append(UsageEvent(turn.usage or Usage(input_tokens=10, output_tokens=5)))
        stop = turn.stop_reason or ("tool_use" if turn.tool_calls else "end_turn")
        events.append(Done(stop))
        return events

    def _check_request(self, request: ProviderRequest) -> None:
        if not request.messages:
            raise FatalError("Mock API 400: messages must not be empty", 400)
        if request.messages[-1].role != "user":
            raise FatalError("Mock API 400: the last message must be from the user", 400)
        violations = validate(request.messages)
        if violations:
            details = "; ".join(f"{v.code}@{v.index}: {v.detail}" for v in violations)
            raise FatalError(f"Mock API 400: invalid message history: {details}", 400)

    def stream(self, request: ProviderRequest, cancel: CancelToken) -> Iterator[StreamEvent]:
        self.requests.append(
            ProviderRequest(
                system=request.system,
                messages=[m.model_copy(deep=True) for m in request.messages],
                tools=list(request.tools),
                max_tokens=request.max_tokens,
            )
        )
        self._check_request(request)
        if not self._script:
            raise AssertionError("MockProvider script exhausted")
        turn = self._script.pop(0)

        if turn.error is not None and turn.error_after_events == 0:
            raise turn.error

        for index, event in enumerate(self._events(turn)):
            if turn.error is not None and index == turn.error_after_events:
                raise turn.error
            if turn.block_after_events is not None and index == turn.block_after_events:
                # Safety timeout so a broken test cannot hang forever.
                if not cancel.wait(10):
                    raise AssertionError("MockProvider blocked for 10s without being cancelled")
                raise Cancelled()
            yield event
