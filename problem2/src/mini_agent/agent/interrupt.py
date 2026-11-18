"""Ctrl+C policy.

Idle at the prompt:            Ctrl+C leaves the program.
While a task is running:       the first Ctrl+C cancels only this task and returns to the prompt;
                               a second Ctrl+C within `window` seconds quits the program.
At a y/n confirmation prompt:  Ctrl+C counts as "no" and cancels the task.

The handler runs in the main thread between two bytecodes of the agent. It must stay tiny: it sets
the CancelToken (all waiting code polls it) and, only where the policy says so, raises. It never
prints, because the console may be in the middle of writing.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from mini_agent.agent.cancel import CancelToken


class QuitRequested(BaseException):
    """Raised by the second quick Ctrl+C. BaseException so no `except Exception` can swallow it."""


class InterruptController:
    def __init__(self, window: float = 2.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.window = window
        self._clock = clock
        self._token: CancelToken | None = None
        self._last_press: float | None = None
        self._prompting = False

    def start_turn(self) -> CancelToken:
        self._token = CancelToken()
        self._last_press = None
        return self._token

    def end_turn(self) -> None:
        self._token = None
        self._last_press = None

    def guarded_input(self, input_fn: Callable[[str], str]) -> Callable[[str], str]:
        """Wrap an input function so that Ctrl+C interrupts the blocked read instead of waiting for Enter."""

        def read(prompt: str) -> str:
            self._prompting = True
            try:
                return input_fn(prompt)
            finally:
                self._prompting = False

        return read

    def handle(self, signum: int | None = None, frame: object | None = None) -> None:
        token = self._token
        if token is None:
            raise KeyboardInterrupt  # idle: leave the program
        now = self._clock()
        if self._last_press is not None and now - self._last_press <= self.window:
            token.cancel()
            raise QuitRequested
        self._last_press = now
        token.cancel()
        if self._prompting:
            raise KeyboardInterrupt  # unblock a pending y/n question

    @contextmanager
    def install(self) -> Iterator[None]:
        if threading.current_thread() is not threading.main_thread():
            yield  # signal handlers can only be installed from the main thread
            return
        previous = signal.signal(signal.SIGINT, self.handle)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)
