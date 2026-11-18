"""Offline token estimation (no tokenizer download, works without network).

A heuristic: about 4 ASCII characters per token and about 1.5 tokens per non-ASCII character
(CJK text). It is only used for the compaction threshold and as a fallback when a provider does
not report usage; real usage numbers always win where available.
"""

from __future__ import annotations

import json
import math

from mini_agent.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from mini_agent.providers.base import ToolSpec

PER_MESSAGE_OVERHEAD = 4
PER_BLOCK_OVERHEAD = 8


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = len(text.encode("ascii", "ignore"))
    other_chars = len(text) - ascii_chars
    return math.ceil(ascii_chars / 4 + other_chars * 1.5)


def estimate_message_tokens(message: Message) -> int:
    total = PER_MESSAGE_OVERHEAD
    for block in message.content:
        if isinstance(block, TextBlock):
            total += estimate_text_tokens(block.text)
        elif isinstance(block, ToolUseBlock):
            args = block.raw_arguments if block.raw_arguments is not None else json.dumps(block.input, ensure_ascii=False)
            total += PER_BLOCK_OVERHEAD + estimate_text_tokens(block.name) + estimate_text_tokens(args)
        elif isinstance(block, ToolResultBlock):
            total += PER_BLOCK_OVERHEAD + estimate_text_tokens(block.content)
    return total


def estimate_request_tokens(system: str, messages: list[Message], tools: list[ToolSpec] | None = None) -> int:
    total = estimate_text_tokens(system)
    total += sum(estimate_message_tokens(m) for m in messages)
    if tools:
        total += estimate_text_tokens(json.dumps([t.model_dump() for t in tools], ensure_ascii=False))
    return total
