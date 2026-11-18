"""Request/response log (JSONL) with secret redaction.

Every line is redacted after serialization, so a key cannot leak through any field: user input,
tool output, error messages. HTTP headers are never logged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mini_agent.messages import ToolUseBlock, Usage
from mini_agent.providers.base import ProviderRequest, StreamResult

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 6

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
]
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/\-]{12,}")


class Redactor:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._literals: list[str] = []
        for secret in secrets:
            self.add(secret)

    def add(self, secret: str | None) -> None:
        if not secret or len(secret) < MIN_SECRET_LENGTH:
            return
        # The JSON-escaped form matters because lines are redacted after serialization.
        for variant in {secret, json.dumps(secret, ensure_ascii=False)[1:-1]}:
            if variant not in self._literals:
                self._literals.append(variant)
        self._literals.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        for literal in self._literals:
            text = text.replace(literal, REDACTED)
        for pattern in _PATTERNS:
            text = pattern.sub(REDACTED, text)
        return _BEARER.sub(lambda m: m.group(1) + REDACTED, text)


class RequestLogger:
    """Appends one JSON object per line; a disabled logger (path=None) is a no-op."""

    def __init__(self, path: str | Path | None, redactor: Redactor | None = None) -> None:
        self.path = Path(path) if path else None
        self.redactor = redactor or Redactor()
        self.write_errors = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data: Any) -> None:
        if self.path is None:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **data}
        line = self.redactor.redact(json.dumps(record, ensure_ascii=False, default=str))
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            self.write_errors += 1  # logging must never break the agent

    def session_start(self, provider: str, model: str, workdir: str, request: ProviderRequest | None = None) -> None:
        data: dict[str, Any] = {"provider": provider, "model": model, "workdir": workdir}
        if request is not None:
            data["system"] = request.system
            data["tools"] = [t.model_dump() for t in request.tools]
        self.log("session_start", **data)

    def request(self, step: int, request: ProviderRequest, provider: str, model: str) -> None:
        self.log(
            "request",
            step=step,
            provider=provider,
            model=model,
            max_tokens=request.max_tokens,
            system=request.system,
            tools=[t.name for t in request.tools],
            messages=[m.model_dump(mode="json") for m in request.messages],
        )

    def response(self, step: int, result: StreamResult, usage: Usage) -> None:
        self.log(
            "response",
            step=step,
            stop_reason=result.stop_reason,
            usage=usage.model_dump(),
            message=result.message.model_dump(mode="json"),
        )

    def retry(self, step: int, attempt: int, delay: float, error: str) -> None:
        self.log("retry", step=step, attempt=attempt, delay=round(delay, 3), error=error)

    def tool(self, step: int, use: ToolUseBlock, content: str, is_error: bool, rejected: bool = False) -> None:
        self.log(
            "tool",
            step=step,
            id=use.id,
            name=use.name,
            input=use.input,
            raw_arguments=use.raw_arguments,
            is_error=is_error,
            rejected=rejected,
            result=content,
        )

    def error(self, step: int, message: str) -> None:
        self.log("error", step=step, message=message)

    def turn_end(self, status: str, usage: Usage, steps: int) -> None:
        self.log("turn_end", status=status, steps=steps, usage=usage.model_dump())
