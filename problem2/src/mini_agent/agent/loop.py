"""The agent loop (hand-written; no agent framework).

One turn: user input -> model -> run the requested tools -> feed every result back -> model ...
until the model answers without tool calls, or a stop condition fires.

Invariants this module upholds (see history.py):
  * every tool_use gets exactly one tool_result, even when the call was invalid, rejected by the
    user, failed, or the turn was interrupted;
  * the history always ends with an assistant message when a turn finishes, so the next user
    message keeps roles alternating;
  * tool results are only ever committed together as ONE user message.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from mini_agent.agent.cancel import CancelToken, Cancelled
from mini_agent.config import AgentConfig
from mini_agent.history import repair, validate
from mini_agent.logging_ import Redactor, RequestLogger
from mini_agent.messages import Message, StopReason, ToolResultBlock, ToolUseBlock, Usage
from mini_agent.providers.base import (
    Provider,
    ProviderRequest,
    StreamEvent,
    StreamResult,
    TextDelta,
    collect,
)
from mini_agent.providers.errors import ProviderError, RetryableError
from mini_agent.providers.retry import RetryPolicy, call_with_retry
from mini_agent.tokens import estimate_message_tokens, estimate_request_tokens
from mini_agent.tools.base import ToolContext, ToolError
from mini_agent.tools.registry import ToolRegistry
from mini_agent.tools.sandbox import Sandbox
from mini_agent.ui import AgentUI

log = logging.getLogger("mini_agent.loop")

TurnStatus = Literal["completed", "max_steps", "too_many_failures", "cancelled", "error"]

REJECTED_MESSAGE = (
    "The user rejected this operation, so it was NOT executed. Do not retry the same operation; "
    "ask the user how they would like to proceed, or choose a different approach."
)
NOT_EXECUTED_CANCELLED = "NOT executed: the user cancelled the turn before this call ran."
NO_OUTPUT = "(no output)"
INTERRUPTED_TURN_NOTE = "(The previous turn ended before a reply was produced.)"


@dataclass
class TurnResult:
    status: TurnStatus
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    error: str | None = None


@dataclass
class _ToolOutcome:
    block: ToolResultBlock
    failed: bool = False
    rejected: bool = False


def invalid_arguments_message(use: ToolUseBlock, stop_reason: StopReason) -> str:
    preview = (use.raw_arguments or "")[:200]
    if stop_reason == "max_tokens":
        return (
            f"The arguments of this {use.name} call were cut off because the reply hit the max_tokens limit, "
            "so the call was NOT executed. Retry with a much smaller payload: for example create the file with a "
            "short write_file call and then extend it with several edit_file calls."
        )
    return (
        f"The arguments of this {use.name} call are not a valid JSON object, so the call was NOT executed. "
        f"Send the arguments as a valid JSON object. Received: {preview!r}"
    )


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        sandbox: Sandbox,
        ui: AgentUI,
        system_prompt: str,
        config: AgentConfig | None = None,
        *,
        logger: RequestLogger | None = None,
        redactor: Redactor | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        history: list[Message] | None = None,
        on_commit: Callable[[Message], None] | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.sandbox = sandbox
        self.ui = ui
        self.system_prompt = system_prompt
        self.config = config or AgentConfig()
        self.redactor = redactor or Redactor()
        self.logger = logger or RequestLogger(None, self.redactor)
        self.retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self.history: list[Message] = history if history is not None else []
        # Called with every message appended to the history (used for session persistence).
        self.on_commit = on_commit

    # ------------------------------------------------------------------ history

    def _commit(self, message: Message) -> None:
        self.history.append(message)
        if self.on_commit is not None:
            self.on_commit(message)

    def _end_dangling_user_message(self) -> None:
        """A previous turn may have ended on a user message (failed or interrupted before a reply).

        Appending an assistant placeholder keeps roles alternating without rewriting history.
        """
        if self.history and self.history[-1].role == "user":
            self._commit(Message.assistant(INTERRUPTED_TURN_NOTE))

    # --------------------------------------------------------------------- turn

    def run_turn(self, user_text: str, cancel: CancelToken | None = None) -> TurnResult:
        cancel = cancel or CancelToken()
        self._end_dangling_user_message()
        self._commit(Message.user(user_text))
        ctx = ToolContext(sandbox=self.sandbox, cancel=cancel, shell_timeout=self.config.shell_timeout)

        usage = Usage()
        steps = 0
        failures = 0
        empty_responses = 0
        outcome = TurnResult(status="completed")
        try:
            while steps < self.config.max_steps:
                steps += 1
                try:
                    result, step_usage = self._call_model(steps, cancel)
                except ProviderError as exc:
                    message = self.redactor.redact(exc.message)
                    self._fail(steps, message)
                    outcome = TurnResult("error", error=message)
                    break
                except Cancelled:
                    raise
                except Exception as exc:
                    log.exception("unexpected error while calling the model")
                    message = self.redactor.redact(f"Unexpected error: {type(exc).__name__}: {exc}")
                    self._fail(steps, message)
                    outcome = TurnResult("error", error=message)
                    break
                usage += step_usage

                message = result.message
                if not message.content:
                    empty_responses += 1
                    if empty_responses > self.config.max_empty_responses:
                        text = "The model returned an empty response repeatedly."
                        self._fail(steps, text)
                        outcome = TurnResult("error", error=text)
                        break
                    self.ui.notice("The model returned an empty response; asking again.", "warn")
                    continue
                empty_responses = 0

                self._commit(message)
                uses = message.tool_uses
                if not uses:
                    if result.stop_reason == "max_tokens":
                        self.ui.notice("The reply was cut off by the max_tokens limit.", "warn")
                    outcome = TurnResult("completed", text=message.text)
                    break

                failures = self._run_tool_round(uses, result.stop_reason, ctx, steps, failures)
                if failures >= self.config.max_consecutive_failures:
                    text = (
                        f"Stopped: {failures} tool calls failed in a row. Tell me how you would like to proceed "
                        "or give a different instruction."
                    )
                    self._commit(Message.assistant(text))
                    self.ui.notice(text, "warn")
                    outcome = TurnResult("too_many_failures", text=text)
                    break
            else:
                text = (
                    f"Stopped: reached the maximum of {self.config.max_steps} steps for one task. "
                    "Say \"continue\" to let me carry on, or give a new instruction."
                )
                self._commit(Message.assistant(text))
                self.ui.notice(text, "warn")
                outcome = TurnResult("max_steps", text=text)
        except Cancelled:
            self.ui.notice("Cancelled.", "warn")
            outcome = TurnResult("cancelled")
        finally:
            self.ui.usage(usage)
            self.logger.turn_end(outcome.status, usage, steps)

        outcome.usage = usage
        outcome.steps = steps
        return outcome

    def _fail(self, step: int, message: str) -> None:
        self.logger.error(step, message)
        self.ui.notice(message, "error")

    # -------------------------------------------------------------------- model

    def _call_model(self, step: int, cancel: CancelToken) -> tuple[StreamResult, Usage]:
        violations = validate(self.history)
        if violations:
            # Safety net only: the loop maintains validity by construction.
            self.logger.log("history_repaired", violations=[f"{v.code}@{v.index}: {v.detail}" for v in violations])
            self.history[:] = repair(self.history)

        request = ProviderRequest(
            system=self.system_prompt,
            messages=list(self.history),
            tools=self.registry.specs(),
            max_tokens=self.config.max_tokens,
        )
        self.logger.request(step, request, self.provider.name, self.provider.model)

        def attempt() -> StreamResult:
            return collect(self.provider.stream(request, cancel), cancel, on_event=self._on_stream_event)

        def on_retry(n: int, delay: float, error: RetryableError) -> None:
            self.ui.text_end()
            text = self.redactor.redact(error.message)
            self.logger.retry(step, n, delay, text)
            self.ui.retry(n, delay, text)

        try:
            result = call_with_retry(attempt, self.retry_policy, cancel, sleep=self._sleep, on_retry=on_retry)
        finally:
            self.ui.text_end()

        usage = result.usage
        if usage is None:
            usage = Usage(
                input_tokens=estimate_request_tokens(request.system, request.messages, request.tools),
                output_tokens=estimate_message_tokens(result.message),
                estimated=True,
            )
        self.logger.response(step, result, usage)
        return result, usage

    def _on_stream_event(self, event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            self.ui.text_delta(event.text)

    # -------------------------------------------------------------------- tools

    def _run_tool_round(
        self,
        uses: list[ToolUseBlock],
        stop_reason: StopReason,
        ctx: ToolContext,
        step: int,
        failures: int,
    ) -> int:
        """Run every requested call in order and commit all results as one user message.

        Returns the updated count of consecutive failures. If anything interrupts the round
        (cancellation, Ctrl+C), the calls without a result get a synthetic one before the
        exception continues, so the history stays valid.
        """
        blocks: list[ToolResultBlock] = []
        try:
            for use in uses:
                ctx.cancel.raise_if_cancelled()
                outcome = self._run_tool(use, stop_reason, ctx, step)
                blocks.append(outcome.block)
                if outcome.failed:
                    failures += 1
                elif not outcome.rejected:
                    failures = 0
        except BaseException:
            done = {b.tool_use_id for b in blocks}
            blocks.extend(
                ToolResultBlock(tool_use_id=u.id, content=NOT_EXECUTED_CANCELLED, is_error=True)
                for u in uses
                if u.id not in done
            )
            self._commit(Message(role="user", content=blocks))
            raise
        self._commit(Message(role="user", content=blocks))
        return failures

    def _run_tool(self, use: ToolUseBlock, stop_reason: StopReason, ctx: ToolContext, step: int) -> _ToolOutcome:
        self.ui.tool_call(use.name, use.input)

        def finish(content: str, *, is_error: bool, failed: bool = False, rejected: bool = False) -> _ToolOutcome:
            content = content if content.strip() else NO_OUTPUT
            self.logger.tool(step, use, content, is_error, rejected)
            self.ui.tool_result(use.name, content, is_error)
            block = ToolResultBlock(tool_use_id=use.id, content=content, is_error=is_error)
            return _ToolOutcome(block, failed=failed, rejected=rejected)

        if not use.arguments_valid:
            return finish(invalid_arguments_message(use, stop_reason), is_error=True, failed=True)
        try:
            tool, args = self.registry.parse(use.name, use.input)
            if tool.requires_confirmation and not self.config.auto_yes:
                preview = tool.preview(args, ctx)
                if not self.ui.confirm(tool.name, preview):
                    return finish(REJECTED_MESSAGE, is_error=True, rejected=True)
        except ToolError as exc:
            return finish(str(exc), is_error=True, failed=True)

        result = self.registry.execute(tool, args, ctx)
        return finish(result.content, is_error=result.is_error, failed=result.is_error)
