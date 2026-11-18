"""File tools: list_files, read_file, write_file, edit_file, grep."""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from mini_agent.tools.base import Tool, ToolArgs, ToolContext, ToolError, ToolResult
from mini_agent.tools.truncate import truncate_chars

IGNORED_DIRS = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea"}
)

MAX_LIST_RESULTS = 200
MAX_READ_LINES = 500
MAX_READ_CHARS = 20_000  # must stay below the registry's last-resort cap so our own continue-hint survives
MAX_LINE_CHARS = 2_000
MAX_GREP_MATCHES = 100
MAX_GREP_LINE_CHARS = 300
MAX_GREP_FILE_BYTES = 20 * 1024 * 1024
MAX_DIFF_CHARS = 8_000


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob (`*`, `?`, `**`) into a regex over '/'-separated relative paths."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matcher(pattern: str):
    """A pattern without '/' matches file names at any depth; otherwise the relative path."""
    regex = glob_to_regex(pattern)
    if "/" in pattern:
        return lambda rel: regex.match(rel) is not None
    return lambda rel: regex.match(rel.rsplit("/", 1)[-1]) is not None


def _walk_files(base: Path) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, path relative to base) for files, pruning ignored dirs in place."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            full = Path(dirpath) / name
            yield full, full.relative_to(base).as_posix()


def _is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"\0" in f.read(8192)


def _read_utf8(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{path.name} is not valid UTF-8 text, so it cannot be edited safely.") from None


def _require_file(ctx: ToolContext, user_path: str) -> Path:
    path = ctx.sandbox.resolve(user_path)
    if path.is_dir():
        raise ToolError(f"{user_path!r} is a directory, not a file. Use list_files to see its contents.")
    if not path.exists():
        raise ToolError(f"File not found: {user_path!r}. Use list_files to check the path.")
    return path


def _diff(old: str, new: str, rel: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=f"a/{rel}", tofile=f"b/{rel}"
    )
    return truncate_chars("".join(lines), MAX_DIFF_CHARS) or "(no changes)"


# ---------------------------------------------------------------- list_files


class ListFilesArgs(ToolArgs):
    path: str = Field(".", description="Directory to list, relative to the working directory.")
    pattern: str | None = Field(
        None,
        description=(
            "Optional glob such as '*.py' or 'src/**/*.ts'. When given, files are searched recursively; "
            "a pattern without '/' matches file names at any depth."
        ),
    )


class ListFilesTool(Tool):
    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = (
        "List the contents of a directory, or find files recursively by glob pattern. "
        "Ignores .git, node_modules, __pycache__, virtualenvs and similar directories."
    )
    params = ListFilesArgs

    def run(self, args: ListFilesArgs, ctx: ToolContext) -> ToolResult:
        base = ctx.sandbox.resolve(args.path)
        if not base.is_dir():
            raise ToolError(f"{args.path!r} is not a directory.")

        if args.pattern is None:
            entries = []
            with os.scandir(base) as it:
                for entry in sorted(it, key=lambda e: e.name):
                    if entry.is_dir() and entry.name in IGNORED_DIRS:
                        continue
                    entries.append(entry.name + ("/" if entry.is_dir() else ""))
        else:
            match = _matcher(args.pattern)
            entries = [rel for _, rel in _walk_files(base) if match(rel)]

        if not entries:
            return ToolResult("No entries found." if args.pattern is None else "No files matched the pattern.")
        shown = entries[:MAX_LIST_RESULTS]
        text = "\n".join(shown)
        if len(entries) > len(shown):
            text += (
                f"\n[Truncated: {len(entries)} entries in total, showing the first {len(shown)}. "
                "Use a narrower path or pattern.]"
            )
        return ToolResult(text)


# ----------------------------------------------------------------- read_file


class ReadFileArgs(ToolArgs):
    path: str = Field(description="File to read, relative to the working directory.")
    start_line: int = Field(1, ge=1, description="First line to read (1-based).")
    end_line: int | None = Field(None, ge=1, description="Last line to read (inclusive). Defaults to the end of the file.")


class ReadFileTool(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        f"Read a text file. Output lines are prefixed with their line numbers. At most {MAX_READ_LINES} lines are "
        "returned per call; use start_line/end_line to read other parts of large files."
    )
    params = ReadFileArgs

    def run(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
        path = _require_file(ctx, args.path)
        if args.end_line is not None and args.end_line < args.start_line:
            raise ToolError(f"end_line ({args.end_line}) must not be smaller than start_line ({args.start_line}).")
        if _is_binary(path):
            raise ToolError(f"{args.path!r} looks like a binary file; it cannot be read as text.")

        wanted_end = args.end_line if args.end_line is not None else args.start_line + MAX_READ_LINES - 1
        wanted_end = min(wanted_end, args.start_line + MAX_READ_LINES - 1)

        out: list[str] = []
        chars = 0
        last_shown = args.start_line - 1
        total = 0
        stopped_by_limit = False
        with open(path, encoding="utf-8", errors="replace") as f:
            for total, line in enumerate(f, start=1):
                if total < args.start_line or total > wanted_end or stopped_by_limit:
                    continue
                text = line.rstrip("\r\n")
                if len(text) > MAX_LINE_CHARS:
                    text = f"{text[:MAX_LINE_CHARS]} [line truncated: {len(text)} characters in total]"
                formatted = f"{total:>6}\t{text}"
                if chars + len(formatted) > MAX_READ_CHARS and out:
                    stopped_by_limit = True
                    continue
                out.append(formatted)
                chars += len(formatted) + 1
                last_shown = total

        if total == 0:
            return ToolResult("(empty file)")
        if args.start_line > total:
            raise ToolError(f"{args.path!r} has only {total} lines; start_line {args.start_line} is past the end.")

        body = "\n".join(out)
        requested_end = args.end_line if args.end_line is not None else total
        cut_by_limit = last_shown < min(requested_end, total)
        if cut_by_limit:
            body += (
                f"\n[Truncated: the file has {total} lines; showing lines {args.start_line}-{last_shown}. "
                f"Call read_file again with start_line={last_shown + 1} to continue.]"
            )
        return ToolResult(body)


# ---------------------------------------------------------------- write_file


class WriteFileArgs(ToolArgs):
    path: str = Field(description="File to create or overwrite, relative to the working directory.")
    content: str = Field(description="Complete new content of the file.")


class WriteFileTool(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create a file or overwrite it completely with the given content. Missing parent directories are created. "
        "To change part of an existing file, prefer edit_file."
    )
    params = WriteFileArgs
    requires_confirmation: ClassVar[bool] = True

    def preview(self, args: WriteFileArgs, ctx: ToolContext) -> str:
        path = ctx.sandbox.resolve(args.path)
        rel = ctx.sandbox.relative(path)
        if path.is_dir():
            raise ToolError(f"{args.path!r} is a directory, not a file.")
        if path.exists():
            try:
                old = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                return f"Overwrite {rel} (existing content is not UTF-8 text)"
            return f"Overwrite {rel}\n{_diff(old, args.content, rel)}"
        lines = args.content.splitlines()
        head = "\n".join(f"+{line}" for line in lines[:40])
        more = f"\n+... ({len(lines) - 40} more lines)" if len(lines) > 40 else ""
        return f"Create {rel} ({len(lines)} lines)\n{head}{more}"

    def run(self, args: WriteFileArgs, ctx: ToolContext) -> ToolResult:
        path = ctx.sandbox.resolve(args.path)
        if path.is_dir():
            raise ToolError(f"{args.path!r} is a directory, not a file.")
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = args.content.encode("utf-8")
        path.write_bytes(data)
        verb = "Overwrote" if existed else "Created"
        return ToolResult(f"{verb} {ctx.sandbox.relative(path)} ({len(args.content.splitlines())} lines, {len(data)} bytes).")


# ----------------------------------------------------------------- edit_file


class EditFileArgs(ToolArgs):
    path: str = Field(description="File to edit, relative to the working directory.")
    old_string: str = Field(description="Exact text to replace. It must occur exactly once in the file.")
    new_string: str = Field(description="Replacement text.")


def _find_all(text: str, needle: str) -> list[int]:
    positions = []
    start = text.find(needle)
    while start != -1:
        positions.append(start)
        start = text.find(needle, start + 1)  # overlapping matches also count as ambiguous
    return positions


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _not_found_message(text: str, old: str, rel: str) -> str:
    hints = []
    stripped = old.strip()
    if stripped and stripped in text:
        hints.append("The text exists if leading/trailing whitespace is ignored, so check indentation and blank lines.")
    if "\r\n" in text and "\n" in old and "\r\n" not in old:
        hints.append("The file uses CRLF line endings but old_string uses LF.")
    hints.append(f"Use read_file to view the exact current content of {rel} and copy old_string from it.")
    return f"old_string was not found in {rel}. " + " ".join(hints)


class EditFileTool(Tool):
    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Edit a file by replacing old_string with new_string. old_string must match the file exactly "
        "(including whitespace) and occur exactly once; include surrounding lines to make it unique."
    )
    params = EditFileArgs
    requires_confirmation: ClassVar[bool] = True

    def _plan(self, args: EditFileArgs, ctx: ToolContext) -> tuple[Path, str, str, int]:
        path = _require_file(ctx, args.path)
        rel = ctx.sandbox.relative(path)
        if args.old_string == "":
            raise ToolError("old_string must not be empty. To create or fully replace a file, use write_file.")
        if args.old_string == args.new_string:
            raise ToolError("old_string and new_string are identical, so the edit would change nothing.")
        text = _read_utf8(path)
        positions = _find_all(text, args.old_string)
        if not positions:
            raise ToolError(_not_found_message(text, args.old_string, rel))
        if len(positions) > 1:
            lines = [str(_line_of(text, p)) for p in positions[:10]]
            more = "..." if len(positions) > 10 else ""
            raise ToolError(
                f"old_string was found {len(positions)} times in {rel} (starting at lines {', '.join(lines)}{more}). "
                "Include more surrounding context in old_string so that it matches exactly once."
            )
        index = positions[0]
        new_text = text[:index] + args.new_string + text[index + len(args.old_string) :]
        return path, text, new_text, _line_of(text, index)

    def preview(self, args: EditFileArgs, ctx: ToolContext) -> str:
        path, old, new, _ = self._plan(args, ctx)
        return f"Edit {ctx.sandbox.relative(path)}\n{_diff(old, new, ctx.sandbox.relative(path))}"

    def run(self, args: EditFileArgs, ctx: ToolContext) -> ToolResult:
        path, _, new_text, line = self._plan(args, ctx)
        path.write_bytes(new_text.encode("utf-8"))
        return ToolResult(f"Edited {ctx.sandbox.relative(path)}: replaced 1 occurrence at line {line}.")


# ---------------------------------------------------------------------- grep


class GrepArgs(ToolArgs):
    pattern: str = Field(description="Regular expression (Python syntax) to search for.")
    path: str = Field(".", description="File or directory to search, relative to the working directory.")
    include: str | None = Field(None, description="Only search files matching this glob, e.g. '*.py'.")
    ignore_case: bool = Field(False, description="Case-insensitive matching.")


class GrepTool(Tool):
    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search file contents with a regular expression. Returns 'path:line:text' for each match. "
        "Skips binary files and ignored directories such as .git and node_modules."
    )
    params = GrepArgs

    def run(self, args: GrepArgs, ctx: ToolContext) -> ToolResult:
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"Invalid regular expression {args.pattern!r}: {exc}") from None

        target = ctx.sandbox.resolve(args.path)
        if not target.exists():
            raise ToolError(f"Path not found: {args.path!r}.")
        include = _matcher(args.include) if args.include else None

        if target.is_file():
            files = [(target, ctx.sandbox.relative(target))]
        else:
            files = ((p, ctx.sandbox.relative(p)) for p, _ in _walk_files(target))

        matches: list[str] = []
        total = 0
        for path, rel in files:
            if include is not None and not include(rel):
                continue
            if not ctx.sandbox.contains(path):  # a symlink pointing outside the working directory
                continue
            try:
                if path.stat().st_size > MAX_GREP_FILE_BYTES or _is_binary(path):
                    continue
                with open(path, encoding="utf-8", errors="replace") as f:
                    for number, line in enumerate(f, start=1):
                        text = line.rstrip("\r\n")
                        if regex.search(text):
                            total += 1
                            if len(matches) < MAX_GREP_MATCHES:
                                if len(text) > MAX_GREP_LINE_CHARS:
                                    text = text[:MAX_GREP_LINE_CHARS] + "..."
                                matches.append(f"{rel}:{number}:{text}")
            except OSError:
                continue

        if not matches:
            return ToolResult("No matches found.")
        result = "\n".join(matches)
        if total > len(matches):
            result += f"\n[Truncated: {total} matches in total, showing the first {len(matches)}. Narrow the pattern or path.]"
        return ToolResult(result)
