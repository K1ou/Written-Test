"""Tool abstractions: argument models, results and the Tool base class."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from mini_agent.agent.cancel import CancelToken
from mini_agent.providers.base import ToolSpec

if TYPE_CHECKING:
    from mini_agent.tools.sandbox import Sandbox


class ToolError(Exception):
    """An expected failure whose message is shown to the model so it can correct itself."""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolContext:
    sandbox: Sandbox
    cancel: CancelToken
    shell_timeout: int = 60


class ToolArgs(BaseModel):
    # Unknown fields are an error; lax type coercion ("5" -> 5) stays on for small models.
    model_config = ConfigDict(extra="forbid")


def clean_schema(schema: Any) -> Any:
    """Strip pydantic noise so every vendor accepts the schema.

    Removes titles, defaults and additionalProperties, and collapses `X | None` into X.
    """
    if isinstance(schema, list):
        return [clean_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        non_null = [s for s in any_of if not (isinstance(s, dict) and s.get("type") == "null")]
        if len(non_null) == 1:
            merged = {k: v for k, v in schema.items() if k != "anyOf"}
            merged.update(non_null[0])
            return clean_schema(merged)

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("title", "default", "additionalProperties"):
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {name: clean_schema(prop) for name, prop in value.items()}
        else:
            cleaned[key] = clean_schema(value)
    return cleaned


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    params: ClassVar[type[ToolArgs]]
    requires_confirmation: ClassVar[bool] = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=clean_schema(self.params.model_json_schema()),
        )

    def preview(self, args: ToolArgs, ctx: ToolContext) -> str:
        """Text shown to the user before confirmation; may raise ToolError for invalid calls."""
        return f"{self.name} {json.dumps(args.model_dump(exclude_none=True), ensure_ascii=False)}"

    @abstractmethod
    def run(self, args: ToolArgs, ctx: ToolContext) -> ToolResult | str: ...
