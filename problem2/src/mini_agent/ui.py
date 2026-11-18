"""Terminal rendering and user confirmation.

The agent only talks to the AgentUI protocol, so tests can plug in a recording UI.
Model text, tool calls and tool results are rendered distinctly; plain ASCII markers are used
so that legacy Windows consoles (GBK) never fail on special characters.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any, Protocol

from rich.console import Console
from rich.text import Text

from mini_agent.messages import Usage

# Arguments worth showing first when summarizing a tool call.
_MAIN_ARGS = ("path", "command", "pattern", "old_string")


class AgentUI(Protocol):
    def text_delta(self, text: str) -> None: ...

    def text_end(self) -> None: ...

    def tool_call(self, name: str, args: dict[str, Any]) -> None: ...

    def tool_result(self, name: str, content: str, is_error: bool) -> None: ...

    def confirm(self, tool_name: str, preview: str) -> bool: ...

    def retry(self, attempt: int, delay: float, error: str) -> None: ...

    def usage(self, usage: Usage) -> None: ...

    def notice(self, message: str, level: str = "info") -> None: ...


def _shorten(value: str, limit: int) -> str:
    value = value.replace("\r", "").replace("\n", "\\n")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def summarize_args(args: dict[str, Any], limit: int = 100) -> str:
    """One-line view of a tool call's arguments, main parameters first."""
    if not args:
        return ""
    ordered = [k for k in _MAIN_ARGS if k in args] + [k for k in args if k not in _MAIN_ARGS]
    parts = []
    for key in ordered:
        value = args[key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if key == "content" and isinstance(value, str):
            text = f"<{len(value)} chars>"
        parts.append(f"{key}={_shorten(text, limit)}")
    return " ".join(parts)


def summarize_result(content: str, limit: int = 150) -> str:
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        return "(no output)"
    summary = _shorten(lines[0], limit)
    return summary + (f"  (+{len(lines) - 1} more lines)" if len(lines) > 1 else "")


class ConsoleUI:
    def __init__(
        self,
        console: Console | None = None,
        *,
        input_fn: Callable[[str], str] | None = None,
        interactive: bool | None = None,
    ) -> None:
        self.console = console or Console(highlight=False)
        self._input = input_fn or input
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self._at_line_start = True

    def _break_line(self) -> None:
        if not self._at_line_start:
            self.console.print()
            self._at_line_start = True

    def text_delta(self, text: str) -> None:
        if not text:
            return
        self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
        self._at_line_start = text.endswith("\n")

    def text_end(self) -> None:
        self._break_line()

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._break_line()
        self.console.print(Text.assemble((f"> {name} ", "bold cyan"), (summarize_args(args), "cyan")))

    def tool_result(self, name: str, content: str, is_error: bool) -> None:
        self._break_line()
        label, style = ("error", "red") if is_error else ("ok", "green")
        self.console.print(Text(f"  {label}: {summarize_result(content)}", style=style))

    def confirm(self, tool_name: str, preview: str) -> bool:
        self._break_line()
        self.console.print(Text(f"--- {tool_name}: requested operation ---", style="bold yellow"))
        for line in preview.splitlines() or [""]:
            style = "green" if line.startswith("+") else "red" if line.startswith("-") else ""
            self.console.print(Text(line, style=style), soft_wrap=True)
        self.console.print(Text("---", style="bold yellow"))
        if not self.interactive:
            self.console.print(Text("Not an interactive terminal and --yes was not given: operation rejected.", style="yellow"))
            return False
        try:
            answer = self._input("Allow this operation? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

    def retry(self, attempt: int, delay: float, error: str) -> None:
        self._break_line()
        self.console.print(Text(f"  request failed ({_shorten(error, 120)}); retry {attempt} in {delay:.1f}s", style="yellow"))

    def usage(self, usage: Usage) -> None:
        self._break_line()
        estimate = " (estimated)" if usage.estimated else ""
        self.console.print(Text(f"tokens: input {usage.input_tokens} / output {usage.output_tokens}{estimate}", style="dim"))

    def notice(self, message: str, level: str = "info") -> None:
        self._break_line()
        style = {"error": "bold red", "warn": "yellow"}.get(level, "dim")
        self.console.print(Text(message, style=style))
