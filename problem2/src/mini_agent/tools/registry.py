"""Tool registry: lookup, argument validation with model-friendly errors, safe execution."""

from __future__ import annotations

import difflib
import logging

from pydantic import ValidationError

from mini_agent.providers.base import ToolSpec
from mini_agent.tools.base import Tool, ToolArgs, ToolContext, ToolError, ToolResult
from mini_agent.tools.fs import EditFileTool, GrepTool, ListFilesTool, ReadFileTool, WriteFileTool
from mini_agent.tools.shell import RunShellTool
from mini_agent.tools.truncate import truncate_chars

log = logging.getLogger("mini_agent.tools")


def format_validation_error(tool: Tool, error: ValidationError) -> str:
    problems = []
    for err in error.errors():
        field = ".".join(str(p) for p in err["loc"]) or "arguments"
        kind = err["type"]
        if kind == "missing":
            problems.append(f"missing required field '{field}'")
        elif kind == "extra_forbidden":
            problems.append(f"unexpected field '{field}'")
        else:
            got = repr(err.get("input"))
            got = got if len(got) <= 60 else got[:57] + "..."
            problems.append(f"field '{field}': {err['msg']} (got {got})")
    schema = tool.params.model_json_schema()
    required = set(schema.get("required", []))
    fields = ", ".join(f"{n}{'*' if n in required else ''}" for n in schema.get("properties", {}))
    return f"Invalid arguments for {tool.name}: " + "; ".join(problems) + f". Valid fields (* = required): {fields}."


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def parse(self, name: str, raw_input: dict) -> tuple[Tool, ToolArgs]:
        """Find the tool and validate its arguments; raise ToolError with a fixable message."""
        tool = self._tools.get(name)
        if tool is None:
            message = f"Unknown tool {name!r}. Available tools: {', '.join(self._tools)}."
            close = difflib.get_close_matches(name, list(self._tools), n=1)
            if close:
                message += f" Did you mean {close[0]!r}?"
            raise ToolError(message)
        try:
            return tool, tool.params.model_validate(raw_input)
        except ValidationError as exc:
            raise ToolError(format_validation_error(tool, exc)) from None

    def execute(self, tool: Tool, args: ToolArgs, ctx: ToolContext) -> ToolResult:
        """Run a tool; failures come back as error results instead of exceptions."""
        try:
            result = tool.run(args, ctx)
        except ToolError as exc:
            return ToolResult(str(exc), is_error=True)
        except Exception as exc:  # a tool bug must never take the whole agent down
            log.exception("tool %s crashed", tool.name)
            return ToolResult(f"Internal error while running {tool.name}: {type(exc).__name__}: {exc}", is_error=True)
        if isinstance(result, str):
            result = ToolResult(result)
        result.content = truncate_chars(result.content)
        return result


def default_registry() -> ToolRegistry:
    return ToolRegistry([ListFilesTool(), ReadFileTool(), WriteFileTool(), EditFileTool(), GrepTool(), RunShellTool()])
