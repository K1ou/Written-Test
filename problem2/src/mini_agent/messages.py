"""Provider-neutral internal message format.

Agent code only ever sees these models; vendor wire formats are converted at the
provider boundary. Tool results live inside user messages, which is the strictest
common denominator of the vendor protocols.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

Role = Literal["user", "assistant"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]

# User-role messages written by the agent itself (not typed by the user) start with this, so that
# compaction does not mistake them for user requests.
AGENT_NOTICE_PREFIX = "[Agent notice]"


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    # Original argument string, kept only when it could not be parsed into an object.
    raw_arguments: str | None = None
    # Opaque vendor-specific data that must be echoed back (e.g. Gemini thought signatures).
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def arguments_valid(self) -> bool:
        return self.raw_arguments is None

    @classmethod
    def from_raw(cls, id: str, name: str, raw_arguments: str) -> ToolUseBlock:
        """Build from a raw JSON argument string without ever storing invalid JSON as input."""
        text = raw_arguments.strip()
        if not text:
            return cls(id=id, name=name, input={})
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return cls(id=id, name=name, input={}, raw_arguments=raw_arguments)
        if not isinstance(parsed, dict):
            return cls(id=id, name=name, input={}, raw_arguments=raw_arguments)
        return cls(id=id, name=name, input=parsed)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


Block = Annotated[TextBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")]


class Message(BaseModel):
    role: Role
    content: list[Block]

    @model_validator(mode="after")
    def _blocks_match_role(self) -> Message:
        for block in self.content:
            if self.role == "assistant" and isinstance(block, ToolResultBlock):
                raise ValueError("assistant messages cannot contain tool_result blocks")
            if self.role == "user" and isinstance(block, ToolUseBlock):
                raise ValueError("user messages cannot contain tool_use blocks")
        return self

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextBlock(text=text)])

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    # True when the provider did not report usage and the numbers are our own estimate.
    estimated: bool = False

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated=self.estimated or other.estimated,
        )
