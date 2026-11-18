"""Provider abstraction.

A provider turns a ProviderRequest (internal messages) into a stream of vendor-neutral
StreamEvents. The agent loop never sees vendor formats: it only consumes events through
`collect`, which assembles them into an internal assistant Message.

Event contract: every tool call is announced with ToolCallStart (a unique id), followed by
zero or more ToolCallArgsDelta fragments (raw JSON text) and a ToolCallEnd. A stream must end
with exactly one Done; a stream that stops without Done is treated as a dropped connection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from mini_agent.agent.cancel import CancelToken, Cancelled
from mini_agent.messages import Message, StopReason, TextBlock, ToolUseBlock, Usage
from mini_agent.providers.errors import RetryableError


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ProviderRequest:
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int = 4096


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStart:
    id: str
    name: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallArgsDelta:
    id: str
    delta: str


@dataclass(frozen=True)
class ToolCallEnd:
    id: str


@dataclass(frozen=True)
class UsageEvent:
    """Cumulative usage for this request; the latest event wins."""

    usage: Usage


@dataclass(frozen=True)
class Done:
    stop_reason: StopReason


StreamEvent = TextDelta | ToolCallStart | ToolCallArgsDelta | ToolCallEnd | UsageEvent | Done


class Provider(Protocol):
    name: str
    model: str

    def stream(self, request: ProviderRequest, cancel: CancelToken) -> Iterator[StreamEvent]: ...


@dataclass
class StreamResult:
    # May have empty content (the model produced nothing); the caller must not append that.
    message: Message
    stop_reason: StopReason
    usage: Usage | None


@dataclass
class _PendingCall:
    id: str
    name: str
    meta: dict[str, Any]
    args: list[str] = field(default_factory=list)
    ended: bool = False


class StreamAccumulator:
    def __init__(self) -> None:
        self._order: list[list[str] | _PendingCall] = []
        self._calls: dict[str, _PendingCall] = {}
        self.usage: Usage | None = None
        self.stop_reason: StopReason | None = None

    @property
    def done(self) -> bool:
        return self.stop_reason is not None

    def feed(self, event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            if self._order and isinstance(self._order[-1], list):
                self._order[-1].append(event.text)
            else:
                self._order.append([event.text])
        elif isinstance(event, ToolCallStart):
            if event.id in self._calls:
                raise ValueError(f"duplicate tool call id in stream: {event.id!r}")
            call = _PendingCall(event.id, event.name, dict(event.meta))
            self._calls[event.id] = call
            self._order.append(call)
        elif isinstance(event, ToolCallArgsDelta):
            self._calls[event.id].args.append(event.delta)
        elif isinstance(event, ToolCallEnd):
            self._calls[event.id].ended = True
        elif isinstance(event, UsageEvent):
            self.usage = event.usage
        elif isinstance(event, Done):
            self.stop_reason = event.stop_reason

    def _build(self, include_call: Callable[[_PendingCall], bool]) -> Message:
        blocks: list[TextBlock | ToolUseBlock] = []
        for item in self._order:
            if isinstance(item, list):
                text = "".join(item)
                if text.strip():
                    blocks.append(TextBlock(text=text))
            elif include_call(item):
                block = ToolUseBlock.from_raw(item.id, item.name, "".join(item.args))
                block.meta = item.meta
                blocks.append(block)
        return Message(role="assistant", content=list(blocks))

    def partial_message(self) -> Message | None:
        """What to keep when a stream is cut short: text so far plus fully received tool calls.

        Tool calls whose arguments were still arriving are dropped, since their JSON is incomplete.
        """
        message = self._build(lambda call: call.ended)
        return message if message.content else None

    def result(self) -> StreamResult:
        if self.stop_reason is None:
            raise RetryableError("The response stream ended before completion (connection dropped?)")
        # After Done every announced call counts as complete; truncated arguments surface
        # as invalid JSON (raw_arguments set) instead of being silently repaired.
        return StreamResult(self._build(lambda call: True), self.stop_reason, self.usage)


def collect(
    stream: Iterator[StreamEvent],
    cancel: CancelToken,
    on_event: Callable[[StreamEvent], None] | None = None,
) -> StreamResult:
    """Drive a provider stream to completion, honouring cancellation between events."""
    acc = StreamAccumulator()
    try:
        for event in stream:
            acc.feed(event)
            if on_event is not None:
                on_event(event)
            if cancel.is_cancelled:
                raise Cancelled()
    except Cancelled as exc:
        exc.partial = acc.partial_message()
        raise
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return acc.result()
