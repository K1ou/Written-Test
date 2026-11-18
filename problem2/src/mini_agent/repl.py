"""Interactive loop: read a task, run one agent turn, repeat.

Slash commands are handled here and never sent to the model. Ctrl+C follows the policy in
agent/interrupt.py: at the prompt it leaves the program; while a task runs it cancels just that
task, and a second press within two seconds quits.
"""

from __future__ import annotations

from collections.abc import Callable

from mini_agent.agent.interrupt import InterruptController, QuitRequested
from mini_agent.agent.loop import Agent
from mini_agent.ui import AgentUI

HELP_TEXT = (
    "Describe a programming task in plain language and press Enter.\n"
    "Ctrl+C while a task is running cancels that task; press it twice quickly to quit.\n"
    "Commands:\n"
    "  /help   show this help\n"
    "  /exit   leave the program (Ctrl+D or Ctrl+C at the prompt works too)"
)
KNOWN_COMMANDS = ("/help", "/exit")


class Repl:
    def __init__(
        self,
        agent: Agent,
        ui: AgentUI,
        session_id: str,
        input_fn: Callable[[str], str] = input,
        controller: InterruptController | None = None,
    ) -> None:
        self.agent = agent
        self.ui = ui
        self.session_id = session_id
        self._input = input_fn
        self.controller = controller or InterruptController()

    def run(self) -> int:
        self.ui.notice("Type a task, /help for help, /exit to leave.")
        try:
            self._loop()
        except KeyboardInterrupt:
            pass  # Ctrl+C while idle: leave the program
        self.ui.notice(f"Session saved. Resume it with: mini-agent --resume {self.session_id}")
        return 0

    def _loop(self) -> None:
        while True:
            try:
                line = self._input("> ")
            except EOFError:
                return
            text = line.strip()
            if not text:
                continue
            if text == "/exit":
                return
            if text == "/help":
                self.ui.notice(HELP_TEXT)
                continue
            if text.startswith("/") and len(text.split()) == 1:
                self.ui.notice(f"Unknown command {text}. Known commands: {', '.join(KNOWN_COMMANDS)}.", "warn")
                continue
            token = self.controller.start_turn()
            try:
                self.agent.run_turn(text, token)
            except QuitRequested:
                self.ui.notice("Quitting.", "warn")
                return
            finally:
                self.controller.end_turn()
