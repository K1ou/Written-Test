from __future__ import annotations

import threading

from mini_agent.messages import Message


class Cancelled(Exception):
    """Raised when the current turn is cancelled by the user."""

    def __init__(self, partial: Message | None = None) -> None:
        super().__init__("cancelled")
        # Whatever assistant content had already been streamed when cancellation hit.
        self.partial = partial


class CancelToken:
    """One token per turn; shared by the streaming loop, tools and retry sleeps."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        """Sleep up to `timeout` seconds; returns True as soon as the token is cancelled."""
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled()
