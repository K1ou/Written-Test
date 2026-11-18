"""Output limits. Every cut tells the model what was omitted and how to get the rest."""

from __future__ import annotations

import locale

MAX_RESULT_CHARS = 30_000


def truncate_chars(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Last-resort cap on any tool result: keep the first `limit` characters."""
    if len(text) <= limit:
        return text
    kept = text[:limit]
    total_lines = text.count("\n") + 1
    shown_lines = kept.count("\n") + 1
    return (
        f"{kept}\n[Truncated: output has {total_lines} lines ({len(text)} characters); "
        f"showing the first {shown_lines} lines. Narrow the request or read it in smaller parts.]"
    )


def decode_output(data: bytes) -> str:
    """Decode process output: UTF-8 first, then the system code page (e.g. GBK on Chinese Windows)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False), errors="replace")


class HeadTailBuffer:
    """Bounded capture of a byte stream: keeps the first and the last bytes, counts the rest.

    Command failures usually show at the end of the output, so the tail matters as much as
    the head. Memory use is bounded no matter how much a process prints.
    """

    def __init__(self, head_bytes: int = 4_000, tail_bytes: int = 8_000) -> None:
        self._head_limit = head_bytes
        self._tail_limit = tail_bytes
        self._head = bytearray()
        self._tail = bytearray()
        self.total_bytes = 0
        self.total_lines = 0

    def write(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        self.total_lines += chunk.count(b"\n")
        room = self._head_limit - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            if len(self._tail) > self._tail_limit:
                del self._tail[: len(self._tail) - self._tail_limit]

    @property
    def truncated(self) -> bool:
        return self.total_bytes > len(self._head) + len(self._tail)

    def render(self) -> str:
        head = decode_output(bytes(self._head))
        if not self.truncated:
            return head + decode_output(bytes(self._tail))
        omitted = self.total_bytes - len(self._head) - len(self._tail)
        tail = decode_output(bytes(self._tail))
        return (
            f"{head}\n[... {omitted} bytes omitted; the output has about {self.total_lines} lines "
            f"and {self.total_bytes} bytes in total. Redirect the command output to a file and use "
            f"read_file or grep to inspect it ...]\n{tail}"
        )
