"""Detects a model that is going around in circles.

Each tool call is recorded as (signature, result digest). The signature is the tool name plus the
arguments as canonical JSON, so key order and spacing do not matter. The RESULT is part of the
record on purpose: re-running the same test command after a code change is legitimate progress
(the output changes), while the same call with the same output is not.

    rule A  the same record several times in a row           read a.py, read a.py, read a.py
    rule B  a cycle of 2..max_period records repeated        read a.py, run tests, read a.py, run tests, ...

Escalation: every further record that keeps the loop going raises the level; any record that breaks
the pattern resets it. Level 1 and 2 are hints added to the tool result, level 3 stops the task.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mini_agent.messages import ToolUseBlock

MAX_KEPT = 60
LABEL_CHARS = 80


@dataclass(frozen=True)
class LoopVerdict:
    level: int  # 1 = hint, 2 = strong hint, 3 = stop
    rule: str  # "same" or "cycle"
    description: str
    repeats: int

    def hint(self) -> str:
        what = f"{self.description} (repeated {self.repeats} times with the same results)"
        if self.level == 1:
            return (
                f"[Loop warning: {what}. Repeating it will not give a different outcome. Change your approach: "
                "use different arguments or another tool, or tell the user what is blocking you.]"
            )
        return (
            f"[STRONG LOOP WARNING: {what}. You are still repeating yourself. Do NOT make this call again. "
            "If you do, the task will be stopped and the user will be asked how to proceed.]"
        )

    def stop_text(self) -> str:
        return (
            f"Stopped: I seem to be stuck in a loop: {self.description}. Further attempts are unlikely to help. "
            "Tell me how you would like to proceed, or give a different instruction."
        )


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).hexdigest()


def _signature(use: ToolUseBlock) -> str:
    arguments = use.raw_arguments if use.raw_arguments is not None else json.dumps(
        use.input, sort_keys=True, ensure_ascii=False, default=str
    )
    return f"{use.name} {arguments}"


def _label(use: ToolUseBlock) -> str:
    main = next((str(use.input[k]) for k in ("path", "command", "pattern") if k in use.input), "")
    text = f"{use.name} {main}".strip()
    return text if len(text) <= LABEL_CHARS else text[: LABEL_CHARS - 3] + "..."


class LoopDetector:
    def __init__(self, threshold: int = 3, max_period: int = 3) -> None:
        self.threshold = max(2, threshold)
        self.max_period = max(2, max_period)
        self._records: list[tuple[str, str]] = []
        self._labels: list[str] = []
        self._level = 0

    def reset(self) -> None:
        self._records.clear()
        self._labels.clear()
        self._level = 0

    def record(self, use: ToolUseBlock, result: str, is_error: bool) -> LoopVerdict | None:
        """Add one finished call; returns a verdict while the loop is going on, None otherwise."""
        self._records.append((_signature(use), _digest(f"{is_error}:{result}")))
        self._labels.append(_label(use))
        del self._records[:-MAX_KEPT], self._labels[:-MAX_KEPT]

        found = self._find()
        if found is None:
            self._level = 0
            return None
        self._level += 1
        rule, description, repeats = found
        return LoopVerdict(min(self._level, 3), rule, description, repeats)

    def _find(self) -> tuple[str, str, int] | None:
        records, t = self._records, self.threshold
        if len(records) >= t and len(set(records[-t:])) == 1:
            repeats = 0
            while repeats < len(records) and records[-1 - repeats] == records[-1]:
                repeats += 1
            return "same", f"calling {self._labels[-1]}", repeats

        for period in range(2, self.max_period + 1):
            need = period * t
            if len(records) < need:
                continue
            window = records[-need:]
            pattern = window[:period]
            if len(set(pattern)) > 1 and all(window[i] == pattern[i % period] for i in range(need)):
                steps = " -> ".join(self._labels[-need : -need + period])
                return "cycle", f"going around in a circle: {steps}", t
        return None
