"""run_shell: execute a command with a timeout, capped output and process-tree termination."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import ClassVar

from pydantic import Field

from mini_agent.agent.cancel import CancelToken
from mini_agent.tools.base import Tool, ToolArgs, ToolContext, ToolResult
from mini_agent.tools.truncate import HeadTailBuffer

IS_WINDOWS = sys.platform == "win32"
MAX_TIMEOUT = 600


def sanitized_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Copy of the environment without API keys, so commands cannot read or print them."""
    source = os.environ if env is None else env
    return {k: v for k, v in source.items() if not k.upper().endswith("API_KEY")}


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate the process and all of its descendants."""
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            timeout=15,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and _group_alive(proc.pid):
                time.sleep(0.05)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


@dataclass
class ShellOutcome:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False


def _drain(stream, buffer: HeadTailBuffer) -> None:
    try:
        while True:
            chunk = stream.read1(65536)
            if not chunk:
                return
            buffer.write(chunk)
    except (OSError, ValueError):
        return


def run_command(command: str, cwd: str, timeout: float, cancel: CancelToken) -> ShellOutcome:
    kwargs: dict = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        env=sanitized_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    out_buf, err_buf = HeadTailBuffer(), HeadTailBuffer()
    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, out_buf), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_buf), daemon=True),
    ]
    for t in readers:
        t.start()

    timed_out = cancelled = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel.is_cancelled:
                cancelled = True
            elif time.monotonic() >= deadline:
                timed_out = True
            if cancelled or timed_out:
                kill_process_tree(proc)
                break
    finally:
        if proc.poll() is None:  # e.g. KeyboardInterrupt while waiting
            kill_process_tree(proc)
        # A background child may still hold the pipes open; wait briefly, then give up on the
        # readers (daemon threads) instead of blocking until that child exits.
        join_deadline = time.monotonic() + 2
        for t in readers:
            t.join(timeout=max(0.0, join_deadline - time.monotonic()))
        # Closing a stream that a reader thread is blocked on would itself block, so only
        # close the streams whose reader has finished.
        for t, stream in zip(readers, (proc.stdout, proc.stderr)):
            if not t.is_alive():
                try:
                    stream.close()
                except OSError:
                    pass

    return ShellOutcome(
        exit_code=None if (timed_out or cancelled) else proc.returncode,
        stdout=out_buf.render(),
        stderr=err_buf.render(),
        timed_out=timed_out,
        cancelled=cancelled,
    )


class RunShellArgs(ToolArgs):
    command: str = Field(min_length=1, description="Shell command to run in the working directory.")
    timeout: int | None = Field(
        None, ge=1, le=MAX_TIMEOUT, description="Timeout in seconds (default 60). The command is killed when it expires."
    )


class RunShellTool(Tool):
    name: ClassVar[str] = "run_shell"
    description: ClassVar[str] = (
        "Run a shell command in the working directory and return its exit code, stdout and stderr. "
        "Commands run without stdin, so they must not wait for interactive input. "
        "Long outputs are cut to the first and last part."
    )
    params = RunShellArgs
    requires_confirmation: ClassVar[bool] = True

    def preview(self, args: RunShellArgs, ctx: ToolContext) -> str:
        return f"$ {args.command}"

    def run(self, args: RunShellArgs, ctx: ToolContext) -> ToolResult:
        timeout = args.timeout or ctx.shell_timeout
        outcome = run_command(args.command, str(ctx.sandbox.root), timeout, ctx.cancel)

        lines: list[str] = []
        if outcome.cancelled:
            lines.append("[Interrupted by the user: the command and its child processes were killed.]")
        elif outcome.timed_out:
            lines.append(f"[Timed out after {timeout}s: the command and its child processes were killed.]")
        else:
            lines.append(f"exit_code: {outcome.exit_code}")
        if outcome.stdout.strip():
            lines.append(f"--- stdout ---\n{outcome.stdout.rstrip()}")
        if outcome.stderr.strip():
            lines.append(f"--- stderr ---\n{outcome.stderr.rstrip()}")
        # A non-zero exit code is information for the model, not a tool failure.
        return ToolResult("\n".join(lines), is_error=outcome.timed_out or outcome.cancelled)
