"""Provider error taxonomy.

RetryableError: worth retrying with backoff (429 rate limit, 5xx, timeouts, dropped connections).
FatalError:     retrying cannot help (401/403/400/404, exhausted quota, ...); report clearly.
ContextLengthExceeded is a FatalError the agent loop reacts to by compacting history once.
API keys must never appear in any message built here.
"""

from __future__ import annotations


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RetryableError(ProviderError):
    def __init__(self, message: str, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message, status_code)
        self.retry_after = retry_after


class FatalError(ProviderError):
    pass


class ContextLengthExceeded(FatalError):
    pass


class RetriesExhausted(ProviderError):
    def __init__(self, attempts: int, last_error: ProviderError) -> None:
        super().__init__(f"Giving up after {attempts} attempts. Last error: {last_error.message}", last_error.status_code)
        self.attempts = attempts
        self.last_error = last_error


_QUOTA_MARKERS = ("insufficient_quota", "exceeded your current quota", "billing", "arrearage")
_CONTEXT_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "range of input length",
    "input token count",
    "exceeds the maximum",
)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header given in seconds; HTTP-date form is ignored."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def from_http_status(
    status: int,
    message: str,
    *,
    code: str | None = None,
    retry_after: float | None = None,
) -> ProviderError:
    probe = f"{code or ''} {message}".lower()

    if status == 429:
        if any(marker in probe for marker in _QUOTA_MARKERS):
            return FatalError(f"Quota exhausted or billing problem (429): {message}", status)
        return RetryableError(f"Rate limited (429): {message}", status, retry_after)
    if status == 408 or status >= 500:
        return RetryableError(f"Server error or timeout ({status}): {message}", status, retry_after)
    if status == 400:
        if any(marker in probe for marker in _CONTEXT_MARKERS):
            return ContextLengthExceeded(f"Context length exceeded (400): {message}", status)
        return FatalError(f"Bad request (400): {message}", status)
    if status == 401:
        return FatalError(f"Authentication failed (401): check that the API key is set and valid. {message}", status)
    if status == 403:
        return FatalError(f"Permission denied (403): {message}", status)
    if status == 404:
        return FatalError(f"Not found (404): check the model name and base URL. {message}", status)
    return FatalError(f"Request failed ({status}): {message}", status)
