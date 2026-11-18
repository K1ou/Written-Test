"""History invariants: validate, repair and normalize.

Invariants (every provider request must satisfy all of them):
  I1  every tool_use has exactly one tool_result, in the user message right after it
  I2  no orphan tool_result
  I3  roles alternate and the first message is from the user
  I4  tool_use ids are unique
  I5  no empty messages and no blank text blocks
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mini_agent.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock

DEFAULT_MISSING_RESULT = "Tool call was not executed: the session was interrupted before a result was recorded."
DEFAULT_LEADING_PLACEHOLDER = "[Earlier conversation omitted]"


@dataclass(frozen=True)
class Violation:
    code: str
    index: int
    detail: str


def _has_blank_text(message: Message) -> bool:
    return any(isinstance(b, TextBlock) and not b.text.strip() for b in message.content)


def validate(messages: list[Message]) -> list[Violation]:
    violations: list[Violation] = []
    seen_ids: set[str] = set()

    for i, msg in enumerate(messages):
        prev = messages[i - 1] if i > 0 else None

        if not msg.content or _has_blank_text(msg):
            violations.append(Violation("I5", i, "message is empty or has a blank text block"))

        if prev is None:
            if msg.role != "user":
                violations.append(Violation("I3", i, "first message must be from the user"))
        elif msg.role == prev.role:
            violations.append(Violation("I3", i, f"two consecutive {msg.role} messages"))

        if msg.role == "assistant":
            for use in msg.tool_uses:
                if use.id in seen_ids:
                    violations.append(Violation("I4", i, f"duplicate tool_use id {use.id!r}"))
                seen_ids.add(use.id)
            if msg.tool_uses:
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                if nxt is None or nxt.role != "user":
                    ids = [u.id for u in msg.tool_uses]
                    violations.append(Violation("I1", i, f"tool_use {ids} has no following user message"))
                else:
                    counts = Counter(r.tool_use_id for r in nxt.tool_results)
                    for use in msg.tool_uses:
                        if counts[use.id] == 0:
                            violations.append(Violation("I1", i, f"tool_use {use.id!r} has no tool_result"))
                        elif counts[use.id] > 1:
                            violations.append(Violation("I1", i, f"tool_use {use.id!r} has multiple tool_results"))
        else:
            expected = {u.id for u in prev.tool_uses} if prev and prev.role == "assistant" else set()
            for res in msg.tool_results:
                if res.tool_use_id not in expected:
                    violations.append(Violation("I2", i, f"orphan tool_result for {res.tool_use_id!r}"))

    return violations


def normalize(messages: list[Message]) -> list[Message]:
    """Merge adjacent same-role messages; tool_result blocks come first in user messages."""
    merged: list[Message] = []
    for msg in messages:
        if merged and merged[-1].role == msg.role:
            merged[-1] = Message(role=msg.role, content=[*merged[-1].content, *msg.content])
        else:
            merged.append(Message(role=msg.role, content=list(msg.content)))

    result: list[Message] = []
    for msg in merged:
        if msg.role == "user":
            results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            others = [b for b in msg.content if not isinstance(b, ToolResultBlock)]
            result.append(Message(role="user", content=[*results, *others]))
        else:
            result.append(msg)
    return result


def _drop_empty(messages: list[Message]) -> list[Message]:
    cleaned: list[Message] = []
    for msg in messages:
        blocks = [b for b in msg.content if not (isinstance(b, TextBlock) and not b.text.strip())]
        if blocks:
            cleaned.append(Message(role=msg.role, content=blocks))
    return cleaned


def _synthetic_results(ids: list[str], text: str) -> list[ToolResultBlock]:
    return [ToolResultBlock(tool_use_id=i, content=text, is_error=True) for i in ids]


def _fix_pairing(messages: list[Message], missing_text: str) -> list[Message]:
    out: list[Message] = []
    seen_ids: set[str] = set()
    pending: list[str] = []  # tool_use ids of out[-1] that still need a result

    for msg in messages:
        if msg.role == "assistant":
            if pending:
                out.append(Message(role="user", content=list(_synthetic_results(pending, missing_text))))
                pending = []
            blocks = []
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    if block.id in seen_ids:
                        continue
                    seen_ids.add(block.id)
                    pending.append(block.id)
                blocks.append(block)
            if blocks:
                out.append(Message(role="assistant", content=blocks))
        else:
            answered: set[str] = set()
            kept: list[ToolResultBlock] = []
            for res in msg.tool_results:
                if res.tool_use_id in pending and res.tool_use_id not in answered:
                    answered.add(res.tool_use_id)
                    kept.append(res)
            missing = [i for i in pending if i not in answered]
            others = [b for b in msg.content if not isinstance(b, ToolResultBlock)]
            content = [*kept, *_synthetic_results(missing, missing_text), *others]
            pending = []
            if content:
                out.append(Message(role="user", content=content))

    if pending:
        out.append(Message(role="user", content=list(_synthetic_results(pending, missing_text))))
    return out


def repair(
    messages: list[Message],
    missing_result_text: str = DEFAULT_MISSING_RESULT,
    leading_placeholder: str = DEFAULT_LEADING_PLACEHOLDER,
) -> list[Message]:
    """Return a history that satisfies every invariant, changing as little as possible.

    Dangling tool_use blocks get a synthetic error result, orphan or duplicate results and
    duplicate tool_use blocks are dropped, empty content is removed and roles are merged
    back into alternation. An already valid history is returned unchanged.
    """
    result = normalize(_drop_empty(messages))
    result = normalize(_fix_pairing(result, missing_result_text))
    if result and result[0].role != "user":
        result.insert(0, Message.user(leading_placeholder))
    return result
