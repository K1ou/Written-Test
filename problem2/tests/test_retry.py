import random

import pytest

from mini_agent.agent.cancel import CancelToken, Cancelled
from mini_agent.messages import Message
from mini_agent.providers.base import ProviderRequest, collect
from mini_agent.providers.errors import FatalError, RetriesExhausted, RetryableError, from_http_status
from mini_agent.providers.mock import MockProvider, MockTurn
from mini_agent.providers.retry import RetryPolicy, call_with_retry


def flaky(failures: int, error: Exception, value="ok"):
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise error
        return value

    return fn, state


def no_jitter(**kw) -> RetryPolicy:
    return RetryPolicy(jitter=0, **kw)


def test_succeeds_after_two_429_with_growing_delays():
    fn, state = flaky(2, from_http_status(429, "slow"))
    sleeps: list[float] = []
    assert call_with_retry(fn, no_jitter(base_delay=1), CancelToken(), sleep=sleeps.append) == "ok"
    assert state["calls"] == 3
    assert sleeps == [1, 2]


def test_delay_is_capped_by_max_delay():
    fn, _ = flaky(5, RetryableError("x"))
    sleeps: list[float] = []
    call_with_retry(fn, no_jitter(max_retries=5, base_delay=1, max_delay=3), CancelToken(), sleep=sleeps.append)
    assert sleeps == [1, 2, 3, 3, 3]


def test_jitter_stays_within_bounds_and_is_reproducible():
    policy = RetryPolicy(base_delay=4, jitter=0.5, rng=random.Random(1))
    delays = [policy.delay_for(0) for _ in range(50)]
    assert all(2 <= d <= 4 for d in delays) and len(set(delays)) > 1


def test_retry_after_header_overrides_shorter_backoff_and_is_clamped():
    sleeps: list[float] = []
    fn, _ = flaky(1, RetryableError("x", 429, retry_after=10))
    call_with_retry(fn, no_jitter(base_delay=1), CancelToken(), sleep=sleeps.append)
    fn, _ = flaky(1, RetryableError("x", 429, retry_after=500))
    call_with_retry(fn, no_jitter(base_delay=1, max_retry_after=60), CancelToken(), sleep=sleeps.append)
    assert sleeps == [10, 60]


def test_5xx_and_network_errors_retry():
    for error in (from_http_status(503, "down"), RetryableError("connection reset")):
        fn, state = flaky(1, error)
        assert call_with_retry(fn, no_jitter(), CancelToken(), sleep=lambda d: None) == "ok"
        assert state["calls"] == 2


@pytest.mark.parametrize("status", [400, 401])
def test_non_retryable_errors_fail_immediately_with_one_call(status):
    fn, state = flaky(10, from_http_status(status, "nope"))
    sleeps: list[float] = []
    with pytest.raises(FatalError) as info:
        call_with_retry(fn, no_jitter(), CancelToken(), sleep=sleeps.append)
    assert state["calls"] == 1 and sleeps == []
    assert str(status) in info.value.message


def test_gives_up_after_max_retries_with_clear_error():
    fn, state = flaky(100, RetryableError("still down", 503))
    with pytest.raises(RetriesExhausted) as info:
        call_with_retry(fn, no_jitter(max_retries=3), CancelToken(), sleep=lambda d: None)
    assert state["calls"] == 4
    assert info.value.attempts == 4 and "still down" in info.value.message


def test_on_retry_callback_reports_attempt_and_delay():
    fn, _ = flaky(2, RetryableError("x"))
    seen = []
    call_with_retry(fn, no_jitter(), CancelToken(), sleep=lambda d: None, on_retry=lambda n, d, e: seen.append((n, d)))
    assert seen == [(1, 1.0), (2, 2.0)]


def test_unexpected_exceptions_are_not_swallowed():
    fn, state = flaky(1, ValueError("bug"))
    with pytest.raises(ValueError):
        call_with_retry(fn, no_jitter(), CancelToken(), sleep=lambda d: None)
    assert state["calls"] == 1


def test_backoff_sleep_is_interrupted_by_cancel():
    cancel = CancelToken()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        cancel.cancel()
        raise RetryableError("x")

    # Default sleep waits on the token, so a 1000s backoff returns immediately once cancelled.
    with pytest.raises(Cancelled):
        call_with_retry(fn, no_jitter(base_delay=1000, max_delay=1000), cancel)
    assert calls["n"] == 1


def test_no_attempt_is_made_when_already_cancelled():
    cancel = CancelToken()
    cancel.cancel()
    fn, state = flaky(0, RetryableError("x"))
    with pytest.raises(Cancelled):
        call_with_retry(fn, no_jitter(), cancel)
    assert state["calls"] == 0


def test_stream_dropped_midway_is_retried_from_scratch_without_duplicated_text():
    provider = MockProvider(
        [
            MockTurn(text="hello wor", error=RetryableError("connection reset"), error_after_events=2),
            MockTurn(text="hello world"),
        ],
        chunk_size=3,
    )
    cancel = CancelToken()
    req = ProviderRequest(system="s", messages=[Message.user("hi")])
    result = call_with_retry(
        lambda: collect(provider.stream(req, cancel), cancel), no_jitter(), cancel, sleep=lambda d: None
    )
    assert provider.calls == 2
    assert result.message.text == "hello world"


def test_stream_that_ends_without_done_is_retried():
    class Truncating:
        name, model = "t", "t"
        attempts = 0

        def stream(self, request, cancel):
            from mini_agent.providers.base import Done, TextDelta

            Truncating.attempts += 1
            yield TextDelta("par")
            if Truncating.attempts > 1:
                yield TextDelta("tial")
                yield Done("end_turn")

    provider = Truncating()
    cancel = CancelToken()
    req = ProviderRequest(system="s", messages=[Message.user("hi")])
    result = call_with_retry(
        lambda: collect(provider.stream(req, cancel), cancel), no_jitter(), cancel, sleep=lambda d: None
    )
    assert Truncating.attempts == 2 and result.message.text == "partial"
