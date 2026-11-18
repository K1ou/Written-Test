"""Context compaction that can never produce an invalid message sequence.

Why this is tricky: every provider rejects a history in which a tool_use has no tool_result, a
tool_result has no tool_use, or roles do not alternate. A naive "drop the oldest N messages" can cut
right between an assistant message with tool calls and the user message with their results.

The rule that makes it safe: the history is first split into ATOMIC UNITS and cuts only ever fall
between units.

    [user text]                           one unit
    [assistant text]                      one unit
    [assistant with tool_use + the user message holding ALL its tool_result blocks]   one unit

Strategy, cheapest first:
  layer 1  shrink old tool output and big tool arguments inside units (structure is untouched)
  layer 2  replace the oldest units by ONE summary user message, keeping a verbatim recent suffix

The summary always contains, written by code and not by the model: every original user request,
every file that was modified successfully, and a note about what is unfinished. A model-written
progress summary is added on top when available.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mini_agent.history import normalize, repair, validate
from mini_agent.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from mini_agent.tokens import estimate_message_tokens

SUMMARY_PREFIX = "[Conversation summary"
SUMMARY_TITLE = (
    "[Conversation summary: the earlier part of this conversation was compacted to fit the context window. "
    "Everything below is a faithful record; messages after it are verbatim.]"
)
OMITTED_OUTPUT = "[Output omitted to save context"
MODIFYING_TOOLS = ("write_file", "edit_file")

MAX_REQUEST_CHARS = 600
MAX_REQUESTS = 40
MAX_FILES = 80
MAX_NOTES_CHARS = 3_000
MAX_ACTIONS = 12
SHRINK_RESULT_THRESHOLD = 200
SHRINK_ARGUMENT_THRESHOLD = 400

_REQUEST_LINE = re.compile(r"^Request \d+(?: \(latest\))?: (.*)$")
_FILE_LINE = re.compile(r"^- (.+)$")


class CompactionError(Exception):
    """Compaction was not possible or would not have produced a valid history; the caller keeps the original."""


Summarizer = Callable[[list[Message]], "str | None"]


@dataclass
class CompactionResult:
    messages: list[Message]
    layer: int  # 0 = nothing to do / not possible, 1 = shrunk only, 2 = summarized
    before_tokens: int
    after_tokens: int
    summarized_messages: int = 0
    llm_summary: bool = False
    shrunk_units: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.layer > 0


# --------------------------------------------------------------------------- units


def split_units(messages: list[Message]) -> list[list[Message]]:
    """Group a VALID history into atomic units (see the module docstring)."""
    units: list[list[Message]] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.role == "assistant" and message.tool_uses:
            if i + 1 >= len(messages):
                raise CompactionError("a tool call at the end of the history has no results yet")
            units.append([message, messages[i + 1]])
            i += 2
        else:
            units.append([message])
            i += 1
    return units


def flatten(units: list[list[Message]]) -> list[Message]:
    return [m for unit in units for m in unit]


# ------------------------------------------------------------------ layer 1: shrink


def _shrink_text(content: str) -> str:
    lines = content.count("\n") + 1
    return f"{OMITTED_OUTPUT}: {lines} lines, {len(content)} characters. Run the tool again if you need it.]"


def _shrink_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    shrunk: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > SHRINK_ARGUMENT_THRESHOLD and not value.startswith("<omitted"):
            shrunk[key] = f"<omitted {len(value)} characters>"
        else:
            shrunk[key] = value
    return shrunk


def shrink_message(message: Message) -> Message:
    """Return the message with big tool results and big tool arguments replaced by short notes.

    Block types, ids, order and is_error flags are untouched, so pairing stays intact.
    """
    blocks = []
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            if len(block.content) > SHRINK_RESULT_THRESHOLD and not block.content.startswith(OMITTED_OUTPUT):
                block = block.model_copy(update={"content": _shrink_text(block.content)})
        elif isinstance(block, ToolUseBlock):
            block = block.model_copy(update={"input": _shrink_arguments(block.input)})
        blocks.append(block)
    return Message(role=message.role, content=blocks)


# ------------------------------------------------------- what must survive: state


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def is_summary_block(block: Any) -> bool:
    return isinstance(block, TextBlock) and block.text.startswith(SUMMARY_PREFIX)


def parse_summary(text: str) -> tuple[list[str], list[str]]:
    """Read the requests and modified files back out of a summary that this module wrote."""
    requests: list[str] = []
    files: list[str] = []
    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower()
            continue
        if section.startswith("original user requests"):
            match = _REQUEST_LINE.match(line)
            if match:
                requests.append(match.group(1))
        elif section.startswith("files modified"):
            match = _FILE_LINE.match(line)
            if match:
                files.append(match.group(1))
    return requests, files


def collect_state(messages: list[Message]) -> tuple[list[str], list[str]]:
    """User requests (oldest first) and successfully modified files, including those in an earlier summary."""
    requests: list[str] = []
    files: list[str] = []
    results: dict[str, ToolResultBlock] = {}
    for message in messages:
        for result in message.tool_results:
            results[result.tool_use_id] = result

    for message in messages:
        if message.role == "user":
            for block in message.content:
                if is_summary_block(block):
                    old_requests, old_files = parse_summary(block.text)
                    requests.extend(old_requests)
                    files.extend(old_files)
                elif isinstance(block, TextBlock) and block.text.strip():
                    requests.append(_one_line(block.text, MAX_REQUEST_CHARS))
        else:
            for use in message.tool_uses:
                result = results.get(use.id)
                path = use.input.get("path")
                if use.name in MODIFYING_TOOLS and isinstance(path, str) and result is not None and not result.is_error:
                    files.append(path)

    unique_files = list(dict.fromkeys(files))
    return requests, unique_files


def _tool_action_line(use: ToolUseBlock, result: ToolResultBlock | None) -> str:
    main = next((str(use.input[k]) for k in ("path", "command", "pattern") if k in use.input), "")
    outcome = "no result" if result is None else ("error" if result.is_error else "ok")
    return f"- {use.name} {_one_line(main, 80)} -> {outcome}".replace("  ", " ")


def rule_based_notes(prefix: list[Message]) -> str:
    """Progress notes that need no model: what was done, and a warning that work may be unfinished."""
    results = {r.tool_use_id: r for m in prefix for r in m.tool_results}
    actions = [_tool_action_line(u, results.get(u.id)) for m in prefix if m.role == "assistant" for u in m.tool_uses]
    lines = [f"{len(actions)} tool call(s) were made in the compacted part. The most recent ones:"]
    lines += actions[-MAX_ACTIONS:] or ["- (none)"]
    lines.append(
        "Unfinished work: not recorded automatically. Treat the latest request above as possibly unfinished and "
        "continue from the verbatim messages that follow."
    )
    return "\n".join(lines)


def build_summary(requests: list[str], files: list[str], notes: str) -> str:
    omitted = max(0, len(requests) - MAX_REQUESTS)
    shown = requests[omitted:]
    lines = [SUMMARY_TITLE, "", "## Original user requests (oldest first)"]
    if omitted:
        lines.append(f"({omitted} older request(s) omitted)")
    for index, request in enumerate(shown, start=omitted + 1):
        latest = " (latest)" if index == len(requests) else ""
        lines.append(f"Request {index}{latest}: {request}")
    lines += ["", "## Files modified so far"]
    lines += [f"- {path}" for path in files[-MAX_FILES:]] or ["(none)"]
    lines += ["", "## Progress and unfinished work", notes.strip()[:MAX_NOTES_CHARS], "", "[End of summary]"]
    return "\n".join(lines)


# ---------------------------------------------------------------------- transcript


def render_transcript(messages: list[Message], max_chars: int = 60_000) -> str:
    """Plain-text rendering of messages for the summarizer (no tool structure, so no pairing rules to break)."""
    parts: list[str] = []
    for message in messages:
        who = "USER" if message.role == "user" else "ASSISTANT"
        for block in message.content:
            if isinstance(block, TextBlock):
                text = block.text if len(block.text) <= 6_000 else block.text[:6_000] + " [...]"
                parts.append(f"{who}: {text}")
            elif isinstance(block, ToolUseBlock):
                arguments = json.dumps(_shrink_arguments(block.input), ensure_ascii=False)
                parts.append(f"ASSISTANT called {block.name} {_one_line(arguments, 300)}")
            elif isinstance(block, ToolResultBlock):
                status = "ERROR" if block.is_error else "result"
                parts.append(f"TOOL {status}: {_one_line(block.content, 300)}")
    text = "\n".join(parts)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n[... middle of the transcript omitted ...]\n{text[-half:]}"


# --------------------------------------------------------------------- the main function


def compact(
    messages: list[Message],
    *,
    target_tokens: int,
    base_tokens: int = 0,
    scale: float = 1.0,
    keep_recent_units: int = 2,
    reserve_tokens: int = 0,
    summarize: Summarizer | None = None,
) -> CompactionResult:
    """Make `messages` fit into `target_tokens` (including `base_tokens` for system prompt and tool schemas).

    Returns the original messages unchanged (layer 0) when nothing can be done. Raises CompactionError
    if the history is invalid or the produced one would be. The result always passes validate().
    """
    violations = validate(messages)
    if violations:
        raise CompactionError("cannot compact an invalid history: " + "; ".join(f"{v.code}: {v.detail}" for v in violations))

    def tokens(msgs: list[Message]) -> int:
        return base_tokens + int(sum(estimate_message_tokens(m) for m in msgs) * scale)

    before = tokens(messages)
    units = split_units(messages)
    keep_recent = max(1, keep_recent_units)
    if before <= target_tokens:
        return CompactionResult(messages, 0, before, before)
    if len(units) <= keep_recent:
        return CompactionResult(messages, 0, before, before, notes=["too few units to compact"])

    # layer 1: shrink everything except the most recent units, oldest first, until it fits
    shrunk_units = 0
    working = [list(unit) for unit in units]
    for index in range(len(working) - keep_recent):
        shrunk = [shrink_message(m) for m in working[index]]
        if shrunk != working[index]:
            working[index] = shrunk
            shrunk_units += 1
            if tokens(flatten(working)) <= target_tokens:
                return _finish(flatten(working), 1, before, tokens, shrunk_units=shrunk_units)

    # layer 2: the oldest units become one summary; the newest units stay verbatim.
    unit_tokens = [sum(int(estimate_message_tokens(m) * scale) for m in unit) for unit in working]
    budget = target_tokens - base_tokens - reserve_tokens
    cut = len(working) - 1  # the last unit is always kept
    used = unit_tokens[cut]
    while cut > 0 and used + unit_tokens[cut - 1] <= budget:
        cut -= 1
        used += unit_tokens[cut]
    if cut == 0:
        # everything fits verbatim once shrunk, but layer 1 alone was not enough: summarize the oldest unit
        cut = 1

    prefix_units, suffix_units = working[:cut], working[cut:]
    prefix = flatten(prefix_units)
    requests, files = collect_state(prefix)

    notes: str | None = None
    if summarize is not None:
        notes = summarize(prefix)
    llm_summary = bool(notes and notes.strip())
    if not llm_summary:
        notes = rule_based_notes(prefix)

    summary = Message.user(build_summary(requests, files, notes or ""))
    result = normalize([summary, *flatten(suffix_units)])
    if validate(result):
        repaired = repair(result)
        if validate(repaired):
            raise CompactionError("the compacted history is invalid and could not be repaired")
        result = repaired
    return _finish(result, 2, before, tokens, summarized=len(prefix), llm_summary=llm_summary, shrunk_units=shrunk_units)


def _finish(
    messages: list[Message],
    layer: int,
    before: int,
    tokens: Callable[[list[Message]], int],
    *,
    summarized: int = 0,
    llm_summary: bool = False,
    shrunk_units: int = 0,
) -> CompactionResult:
    problems = validate(messages)
    if problems:
        raise CompactionError("the compacted history is invalid: " + "; ".join(f"{v.code}: {v.detail}" for v in problems))
    return CompactionResult(
        messages,
        layer,
        before,
        tokens(messages),
        summarized_messages=summarized,
        llm_summary=llm_summary,
        shrunk_units=shrunk_units,
    )
