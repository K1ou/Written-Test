from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(Exception):
    """A configuration problem the user can fix; shown without a traceback."""


@dataclass
class AgentConfig:
    max_steps: int = 30
    # Stop the turn after this many tool failures in a row (user rejections do not count).
    max_consecutive_failures: int = 5
    # Retries of an empty model response before the turn fails.
    max_empty_responses: int = 2
    max_tokens: int = 4096
    shell_timeout: int = 60
    # Skip the y/n confirmation for write_file, edit_file and run_shell.
    auto_yes: bool = False
    # Context management (see agent/compaction.py). The window is the model's context size in tokens.
    context_window: int = 128_000
    # Compaction starts when the estimated request reaches this share of the window ...
    compact_threshold: float = 0.8
    # ... and shrinks the request to about this share.
    compact_target: float = 0.5
    # Ask the model for a progress summary; when False (or when that call fails) a rule-based one is used.
    llm_summary: bool = True
    # The newest this-many units are never shrunk or summarized.
    keep_recent_units: int = 2
    # Tool calls written as plain text are not executed; the model is told to use the tool interface,
    # at most this many times per task.
    max_text_tool_corrections: int = 2
    # Loop detection (see agent/loop_detector.py): the same call with the same result this many times in a
    # row, or a cycle of 2..loop_max_period calls repeated this many rounds.
    loop_threshold: int = 3
    loop_max_period: int = 3


class Settings(BaseSettings):
    """Provider settings from environment variables and an optional `.env` file.

    API keys are SecretStr, so they print as ********** and never end up in a repr or a log by accident.
    Empty values (like `GEMINI_MODEL=` in .env.example) count as not set.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    dashscope_api_key: SecretStr | None = None
    qwen_base_url: str | None = None
    qwen_model: str = "qwen-plus"
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    mini_agent_provider: str = "qwen"
    mini_agent_home: Path | None = None
    mini_agent_context_window: int = 128_000

    @property
    def state_dir(self) -> Path:
        """Where sessions and logs live: outside the working directory, so tools cannot touch them."""
        return self.mini_agent_home or Path.home() / ".mini-agent"

    def secret_values(self) -> list[str]:
        keys = (self.dashscope_api_key, self.gemini_api_key)
        return [k.get_secret_value() for k in keys if k is not None]
