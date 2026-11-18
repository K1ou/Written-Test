from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from mini_agent.agent.cancel import CancelToken, Cancelled
from mini_agent.providers.errors import RetriesExhausted, RetryableError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_retries: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    # A delay is drawn from [d * (1 - jitter), d]; 0 disables randomness.
    jitter: float = 0.5
    # Server-provided Retry-After is honoured but clamped to this many seconds.
    max_retry_after: float = 60.0
    rng: random.Random = field(default_factory=random.Random)

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before retry number `attempt` (0-based): exponential, capped, jittered."""
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        delay = ceiling * (1 - self.jitter * self.rng.random())
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.max_retry_after))
        return delay


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    cancel: CancelToken,
    *,
    sleep: Callable[[float], None] | None = None,
    on_retry: Callable[[int, float, RetryableError], None] | None = None,
) -> T:
    """Run `fn`, retrying RetryableError with backoff.

    `fn` must start a fresh attempt each call (for streaming: discard any partial output of
    the failed attempt). FatalError and any non-provider exception propagate immediately.
    Backoff sleeps end early with Cancelled when the token is cancelled.
    """
    attempt = 0
    while True:
        cancel.raise_if_cancelled()
        try:
            return fn()
        except RetryableError as error:
            if attempt >= policy.max_retries:
                raise RetriesExhausted(attempt + 1, error) from error
            delay = policy.delay_for(attempt, error.retry_after)
            if on_retry is not None:
                on_retry(attempt + 1, delay, error)
            if sleep is not None:
                sleep(delay)
            elif cancel.wait(delay):
                raise Cancelled() from None
            attempt += 1
