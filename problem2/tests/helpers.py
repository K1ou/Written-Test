"""Shared test helpers: a recording UI and an agent factory wired to the mock provider."""

from __future__ import annotations

from pathlib import Path

from mini_agent.agent.loop import Agent
from mini_agent.agent.prompt import build_system_prompt
from mini_agent.config import AgentConfig
from mini_agent.logging_ import RequestLogger
from mini_agent.messages import Usage
from mini_agent.providers.mock import MockProvider, MockTurn
from mini_agent.providers.retry import RetryPolicy
from mini_agent.tools.registry import default_registry
from mini_agent.tools.sandbox import Sandbox


class RecordingUI:
    def __init__(self, answers: list[bool] | None = None, default: bool = True) -> None:
        self.answers = list(answers or [])
        self.default = default
        self.text = ""
        self.tool_calls: list[tuple[str, dict]] = []
        self.tool_results: list[tuple[str, str, bool]] = []
        self.confirms: list[tuple[str, str]] = []
        self.retries: list[tuple[int, float, str]] = []
        self.usages: list[Usage] = []
        self.notices: list[tuple[str, str]] = []

    def text_delta(self, text):
        self.text += text

    def text_end(self):
        pass

    def tool_call(self, name, args):
        self.tool_calls.append((name, args))

    def tool_result(self, name, content, is_error):
        self.tool_results.append((name, content, is_error))

    def confirm(self, tool_name, preview):
        self.confirms.append((tool_name, preview))
        return self.answers.pop(0) if self.answers else self.default

    def retry(self, attempt, delay, error):
        self.retries.append((attempt, delay, error))

    def usage(self, usage):
        self.usages.append(usage)

    def notice(self, message, level="info"):
        self.notices.append((level, message))


def make_agent(
    script: list[MockTurn],
    workdir: Path,
    *,
    ui: RecordingUI | None = None,
    config: AgentConfig | None = None,
    logger: RequestLogger | None = None,
    redactor=None,
    max_retries: int = 4,
):
    ui = ui or RecordingUI()
    provider = MockProvider(script)
    sleeps: list[float] = []
    agent = Agent(
        provider,
        default_registry(),
        Sandbox(workdir),
        ui,
        build_system_prompt(str(workdir)),
        config or AgentConfig(),
        logger=logger,
        redactor=redactor,
        retry_policy=RetryPolicy(max_retries=max_retries, jitter=0),
        sleep=sleeps.append,
    )
    agent.sleeps = sleeps
    return agent, provider, ui
