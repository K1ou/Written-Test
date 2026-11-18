"""Command-line entry point: wires settings, provider, sandbox, session, logging and the REPL together."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import typer

from mini_agent.agent.interrupt import InterruptController
from mini_agent.agent.loop import Agent
from mini_agent.agent.prompt import build_system_prompt
from mini_agent.config import AgentConfig, ConfigError, Settings
from mini_agent.logging_ import Redactor, RequestLogger
from mini_agent.providers.base import Provider
from mini_agent.providers.factory import PROVIDERS, build_provider
from mini_agent.repl import Repl
from mini_agent.session import SessionError, SessionStore, new_meta
from mini_agent.tools.registry import default_registry
from mini_agent.tools.sandbox import Sandbox
from mini_agent.ui import AgentUI, ConsoleUI

app = typer.Typer(add_completion=False, help="Mini Code Agent: describe a task, the agent edits files and runs commands for you.")


@dataclass
class RunOptions:
    workdir: str | None = None
    provider: str | None = None
    model: str | None = None
    yes: bool = False
    resume: str | None = None
    max_steps: int = 30
    log_file: str | None = None


def _same_dir(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def run(
    options: RunOptions,
    *,
    settings: Settings | None = None,
    provider: Provider | None = None,
    ui: AgentUI | None = None,
    input_fn: Callable[[str], str] = input,
    controller: InterruptController | None = None,
) -> int:
    """Run the interactive agent; returns the process exit code. Fixable problems exit with 2."""
    controller = controller or InterruptController()
    guarded_input = controller.guarded_input(input_fn)
    ui = ui or ConsoleUI(input_fn=guarded_input)
    try:
        return _run(options, settings or Settings(), provider, ui, guarded_input, controller)
    except (ConfigError, SessionError, ValueError) as exc:
        ui.notice(f"Error: {exc}", "error")
        return 2


def _run(
    options: RunOptions,
    settings: Settings,
    provider: Provider | None,
    ui: AgentUI,
    input_fn: Callable[[str], str],
    controller: InterruptController,
) -> int:
    store = SessionStore(settings.state_dir)
    history = []

    if options.resume:
        session, history, report = store.resume(options.resume)
        saved = session.meta
        if options.workdir and not _same_dir(options.workdir, saved.workdir):
            raise ValueError(
                f"Session {saved.session_id} belongs to the working directory {saved.workdir}, "
                f"but --workdir is {options.workdir}. Omit --workdir to continue where you left off."
            )
        workdir = saved.workdir
        provider = provider or build_provider(options.provider or saved.provider, options.model or saved.model, settings)
        if report.dropped_partial_line:
            ui.notice("The last line of the session file was incomplete (the program probably crashed) and was dropped.", "warn")
        if report.interrupted_calls:
            ui.notice(
                f"{report.interrupted_calls} tool call(s) had no recorded result; they were marked as interrupted "
                "with unknown side effects.",
                "warn",
            )
        sandbox = Sandbox(workdir)
    else:
        sandbox = Sandbox(options.workdir or os.getcwd())
        provider = provider or build_provider(options.provider, options.model, settings)
        session = store.create(new_meta(store.new_id(), str(sandbox.root), provider.name, provider.model))

    redactor = Redactor(settings.secret_values())
    log_path = options.log_file or str(settings.state_dir / "logs" / f"{session.id}.log")
    logger = RequestLogger(log_path, redactor)
    agent = Agent(
        provider,
        default_registry(),
        sandbox,
        ui,
        build_system_prompt(str(sandbox.root)),
        AgentConfig(max_steps=options.max_steps, auto_yes=options.yes),
        logger=logger,
        redactor=redactor,
        history=history,
        on_commit=session.append,
    )
    logger.session_start(provider.name, provider.model, str(sandbox.root))
    ui.notice(
        f"Session {session.id} | {provider.name}/{provider.model} | workdir {sandbox.root}"
        + (f" | resumed with {len(history)} messages" if options.resume else "")
    )
    ui.notice(f"Log file: {log_path}")
    with controller.install():
        return Repl(agent, ui, session.id, input_fn, controller).run()


def _make_console_safe() -> None:
    # A legacy console (for example GBK on Chinese Windows) must not crash on characters it cannot show.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@app.command()
def main(
    workdir: Annotated[str | None, typer.Option(help="Working directory the agent may access (default: current directory)")] = None,
    provider: Annotated[str | None, typer.Option(help=f"Model provider: {' or '.join(PROVIDERS)} (default: MINI_AGENT_PROVIDER or qwen)")] = None,
    model: Annotated[str | None, typer.Option(help="Model name (default: QWEN_MODEL / GEMINI_MODEL)")] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation before write_file, edit_file and run_shell")] = False,
    resume: Annotated[str | None, typer.Option(help="Continue the saved session with this id")] = None,
    max_steps: Annotated[int, typer.Option(min=1, help="Maximum model calls for one task")] = 30,
    log_file: Annotated[str | None, typer.Option(help="Write the request/response log here (default: <state dir>/logs/<session id>.log)")] = None,
) -> None:
    _make_console_safe()
    code = run(RunOptions(workdir, provider, model, yes, resume, max_steps, log_file))
    raise typer.Exit(code)
