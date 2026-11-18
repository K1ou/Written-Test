from __future__ import annotations

from dataclasses import dataclass


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
