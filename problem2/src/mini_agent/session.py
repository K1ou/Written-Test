"""Session persistence: one JSONL file per session, appended to message by message.

File layout: line 1 is a `meta` record, every following line is one `message` record.
Each append is flushed and fsynced, so after a crash the file contains every committed message and
at most one half-written last line. Because the agent commits an assistant message (with its
tool_use blocks) BEFORE running the tools, a crash during a tool leaves a dangling tool_use in
the file; resuming repairs it with an explicit "interrupted, side effects unknown" result.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from mini_agent.history import repair
from mini_agent.messages import Message

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

RESUME_MISSING_RESULT = (
    "The program exited while this tool call was running. It may have been fully or partly executed, "
    "so its side effects are unknown. Check the current state (for example with read_file or list_files) "
    "before repeating it."
)


class SessionError(Exception):
    """A problem with a session file that the user can understand and act on."""


class SessionMeta(BaseModel):
    type: Literal["meta"] = "meta"
    session_id: str
    workdir: str
    provider: str
    model: str
    created_at: str


@dataclass
class ResumeReport:
    dropped_partial_line: bool = False
    interrupted_calls: int = 0
    rewritten: bool = False


def _fsync_write(path: Path, data: bytes, mode: str) -> None:
    with open(path, mode) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _encode(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _message_record(message: Message) -> dict[str, Any]:
    return {"type": "message", "message": message.model_dump(mode="json")}


class Session:
    def __init__(self, path: Path, meta: SessionMeta) -> None:
        self.path = path
        self.meta = meta

    @property
    def id(self) -> str:
        return self.meta.session_id

    def append(self, message: Message) -> None:
        _fsync_write(self.path, _encode(_message_record(message)), "ab")


def _parse_record(chunk: bytes) -> dict[str, Any]:
    record = json.loads(chunk.decode("utf-8"))
    if not isinstance(record, dict) or record.get("type") not in ("meta", "message"):
        raise ValueError("not a session record")
    if record["type"] == "message":
        Message.model_validate(record["message"])
    else:
        SessionMeta.model_validate(record)
    return record


class SessionStore:
    def __init__(self, state_dir: Path) -> None:
        self.dir = Path(state_dir) / "sessions"

    @staticmethod
    def new_id() -> str:
        return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    def _path(self, session_id: str) -> Path:
        if not SESSION_ID_RE.match(session_id):
            raise SessionError(f"Invalid session id {session_id!r}: use letters, digits, '-' and '_' only.")
        return self.dir / f"{session_id}.jsonl"

    def recent_ids(self, limit: int = 5) -> list[str]:
        if not self.dir.is_dir():
            return []
        files = sorted(self.dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files[:limit]]

    def create(self, meta: SessionMeta) -> Session:
        path = self._path(meta.session_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        _fsync_write(path, _encode(meta.model_dump(mode="json")), "xb")
        return Session(path, meta)

    def resume(
        self, session_id: str, missing_result_text: str = RESUME_MISSING_RESULT
    ) -> tuple[Session, list[Message], ResumeReport]:
        """Load a session and return a history that is valid to send to any provider.

        A half-written last line is dropped; tool calls without a result get a synthetic
        "interrupted" result. If anything had to change, the file is rewritten atomically so the
        next append cannot end up glued to a broken line.
        """
        path = self._path(session_id)
        if not path.is_file():
            recent = self.recent_ids()
            hint = f" Recent sessions: {', '.join(recent)}." if recent else " There are no saved sessions."
            raise SessionError(f"Session {session_id!r} was not found in {self.dir}.{hint}")

        chunks = path.read_bytes().split(b"\n")
        tail = chunks.pop()  # b"" when the file ends with a newline
        report = ResumeReport()
        needs_rewrite = False

        records: list[dict[str, Any]] = []
        for number, chunk in enumerate(chunks, start=1):
            try:
                records.append(_parse_record(chunk))
            except (ValueError, ValidationError):
                raise SessionError(
                    f"Session file {path.name} is corrupted at line {number}; it cannot be resumed safely."
                ) from None
        if tail.strip():
            needs_rewrite = True
            try:
                records.append(_parse_record(tail))
            except (ValueError, ValidationError):
                report.dropped_partial_line = True

        if not records or records[0]["type"] != "meta" or any(r["type"] == "meta" for r in records[1:]):
            raise SessionError(f"Session file {path.name} has no valid header; it cannot be resumed.")
        meta = SessionMeta.model_validate(records[0])
        loaded = [Message.model_validate(r["message"]) for r in records[1:]]

        repaired = repair(loaded, missing_result_text=missing_result_text)
        used = {u.id for m in loaded for u in m.tool_uses}
        answered = {r.tool_use_id for m in loaded for r in m.tool_results}
        report.interrupted_calls = len(used - answered)
        if repaired != loaded:
            needs_rewrite = True

        session = Session(path, meta)
        if needs_rewrite:
            self._rewrite(path, meta, repaired)
            report.rewritten = True
        return session, repaired, report

    @staticmethod
    def _rewrite(path: Path, meta: SessionMeta, messages: list[Message]) -> None:
        data = _encode(meta.model_dump(mode="json")) + b"".join(_encode(_message_record(m)) for m in messages)
        tmp = path.with_name(path.name + ".tmp")
        _fsync_write(tmp, data, "wb")
        os.replace(tmp, path)


def new_meta(session_id: str, workdir: str, provider: str, model: str) -> SessionMeta:
    return SessionMeta(
        session_id=session_id,
        workdir=workdir,
        provider=provider,
        model=model,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
