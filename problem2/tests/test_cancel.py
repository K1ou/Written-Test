"""4.2 interruption and cancellation: cancel at every stage, and the history must stay valid."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest
from helpers import RecordingUI, make_agent

from mini_agent.agent.cancel import CancelToken, Cancelled
from mini_agent.agent.interrupt import InterruptController, QuitRequested
from mini_agent.agent.loop import CANCEL_NOTE_EMPTY, CANCEL_NOTE_PARTIAL, CANCEL_NOTE_TOOLS
from mini_agent.cli import RunOptions, run
from mini_agent.config import Settings
from mini_agent.history import validate
from mini_agent.messages import Message, ToolResultBlock
from mini_agent.providers.base import ProviderRequest, collect
from mini_agent.providers.errors import RetryableError
from mini_agent.providers.mock import MockProvider, MockToolCall, MockTurn
from mini_agent.session import SessionStore
from mini_agent.tools.shell import IS_WINDOWS

PY = f'"{sys.executable}"'
POSIX_ONLY = pytest.mark.skipif(IS_WINDOWS, reason="needs real SIGINT delivery to a child process")


def cancel_after(token, seconds):
    timer = threading.Timer(seconds, token.cancel)
    timer.start()
    return timer


def pid_alive(pid):
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(predicate, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def history_roles(agent):
    return [m.role for m in agent.history]


# ------------------------------------------------------------ the token


def test_token_wait_returns_early_on_cancel_and_full_time_otherwise():
    token = CancelToken()
    started = time.monotonic()
    assert token.wait(0.25) is False and time.monotonic() - started >= 0.2
    cancel_after(token, 0.1)
    started = time.monotonic()
    assert token.wait(30) is True and time.monotonic() - started < 3


# ------------------------------------------- cancelling while the model streams


def test_cancel_while_the_provider_is_silent_returns_quickly(workdir):
    # A provider blocked on the network (no events at all) must not delay the cancellation.
    provider = MockProvider([MockTurn(text="never delivered", block_after_events=0)])
    token = CancelToken()
    request = ProviderRequest(system="s", messages=[Message.user("hi")])
    timer = cancel_after(token, 0.2)
    started = time.monotonic()
    with pytest.raises(Cancelled) as info:
        collect(provider.stream(request, token), token)
    timer.join()
    assert time.monotonic() - started < 3 and info.value.partial is None


def test_a_hung_stream_that_never_yields_is_abandoned_on_cancel():
    release = threading.Event()

    def hung():
        release.wait(30)  # a read that never returns
        yield from ()

    token = CancelToken()
    timer = cancel_after(token, 0.2)
    started = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            collect(hung(), token)
        assert time.monotonic() - started < 3
    finally:
        release.set()
        timer.join()


def test_a_connection_error_caused_by_the_cancel_is_reported_as_a_cancel_not_a_network_error():
    token = CancelToken()

    def stream():
        yield from ()
        token.cancel()
        raise RetryableError("connection closed by the client")

    with pytest.raises(Cancelled):
        collect(stream(), token)


def test_without_a_cancel_the_same_error_is_still_a_network_error():
    def stream():
        yield from ()
        raise RetryableError("connection reset")

    with pytest.raises(RetryableError):
        collect(stream(), CancelToken())


def test_mid_stream_cancel_keeps_the_streamed_text_and_ends_with_an_assistant_note(workdir):
    ui = RecordingUI()
    script = [MockTurn(text="The answer has several parts: first", block_after_events=3), MockTurn(text="fresh start")]
    agent, provider, _ = make_agent(script, workdir, ui=ui)
    token = CancelToken()
    timer = cancel_after(token, 0.3)
    result = agent.run_turn("explain", token)
    timer.join()
    assert result.status == "cancelled"
    assert history_roles(agent) == ["user", "assistant"]
    kept = agent.history[1].text
    assert kept.startswith("The answer has") and kept.endswith(CANCEL_NOTE_PARTIAL)
    assert validate(agent.history) == []
    assert agent.run_turn("never mind, do this instead").status == "completed"  # strict mock accepts the next request
    assert [m.role for m in provider.requests[1].messages] == ["user", "assistant", "user"]


def test_cancel_in_the_middle_of_a_tool_call_drops_the_incomplete_call(workdir):
    call = MockToolCall("write_file", {"path": "big.txt", "content": "x" * 300})
    script = [MockTurn(text="Writing now. ", tool_calls=[call], block_after_events=6), MockTurn(text="ok")]
    agent, provider, _ = make_agent(script, workdir)
    token = CancelToken()
    cancel_after(token, 0.3)
    agent.run_turn("write a big file", token)
    assert not (workdir / "big.txt").exists()
    assistant = agent.history[1]
    assert assistant.tool_uses == [] and assistant.text.startswith("Writing now.")
    assert validate(agent.history) == []
    assert agent.run_turn("try again smaller").status == "completed"


def test_cancel_before_any_output_leaves_a_note_not_a_dangling_user_message(workdir):
    agent, provider, _ = make_agent([MockTurn(text="x", block_after_events=0), MockTurn(text="ok")], workdir)
    token = CancelToken()
    cancel_after(token, 0.2)
    agent.run_turn("hello", token)
    assert history_roles(agent) == ["user", "assistant"] and agent.history[1].text == CANCEL_NOTE_EMPTY
    assert agent.run_turn("hello again").status == "completed"


def test_cancel_during_a_retry_backoff_wait_returns_promptly(workdir):
    script = [MockTurn(error=RetryableError("Rate limited (429)", 429)) for _ in range(3)]
    agent, provider, ui = make_agent(script, workdir)
    agent._sleep = None  # use the real, cancellable wait
    agent.retry_policy.base_delay = 1000
    agent.retry_policy.max_delay = 1000
    token = CancelToken()
    cancel_after(token, 0.3)
    started = time.monotonic()
    result = agent.run_turn("go", token)
    assert result.status == "cancelled" and time.monotonic() - started < 5 and provider.calls == 1
    assert validate(agent.history) == []


# --------------------------------------------- cancelling while tools run


def test_three_tool_calls_cancelled_during_the_second_every_call_still_gets_a_result(workdir, tmp_path):
    marker = workdir / "third.txt"
    slow = f'{PY} -c "import time; time.sleep(60)"'
    script = [
        MockTurn(
            tool_calls=[
                MockToolCall("write_file", {"path": "first.txt", "content": "one"}),
                MockToolCall("run_shell", {"command": slow}),
                MockToolCall("write_file", {"path": "third.txt", "content": "three"}),
            ]
        ),
        MockTurn(text="continuing after the cancel"),
    ]
    agent, provider, _ = make_agent(script, workdir)
    token = CancelToken()
    cancel_after(token, 1.5)
    started = time.monotonic()
    result = agent.run_turn("do three things", token)
    assert result.status == "cancelled" and time.monotonic() - started < 15

    results = agent.history[2].tool_results
    assert [r.tool_use_id for r in results] == [u.id for u in agent.history[1].tool_uses]
    assert [r.is_error for r in results] == [False, True, True]
    assert "Created first.txt" in results[0].content
    assert "Interrupted by the user" in results[1].content
    assert "NOT executed" in results[2].content and not marker.exists()
    assert history_roles(agent) == ["user", "assistant", "user", "assistant"]
    assert agent.history[3].text == CANCEL_NOTE_TOOLS and validate(agent.history) == []

    assert agent.run_turn("what state are we in?").status == "completed"  # no 400 from the strict mock


def test_cancel_kills_the_whole_process_tree_started_by_run_shell(workdir):
    (workdir / "child.py").write_text("import os, time\nopen('child.pid', 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    (workdir / "parent.py").write_text("import subprocess, sys, time\nsubprocess.Popen([sys.executable, 'child.py'])\ntime.sleep(60)\n")
    script = [MockTurn(tool_calls=[MockToolCall("run_shell", {"command": f"{PY} parent.py"})]), MockTurn(text="done")]
    agent, _, _ = make_agent(script, workdir)
    token = CancelToken()

    def cancel_when_grandchild_runs():
        wait_for(lambda: (workdir / "child.pid").exists(), 30)
        time.sleep(0.3)
        token.cancel()

    threading.Thread(target=cancel_when_grandchild_runs, daemon=True).start()
    result = agent.run_turn("start the slow job", token)
    assert result.status == "cancelled"
    pid = int((workdir / "child.pid").read_text())
    assert wait_for(lambda: not pid_alive(pid), 10), "the grandchild process survived the cancellation"
    assert validate(agent.history) == []


def test_session_saved_after_a_cancel_resumes_into_a_valid_history(tmp_path, workdir):
    store = SessionStore(tmp_path / "state")
    from mini_agent.session import new_meta

    session = store.create(new_meta("s1", str(workdir), "mock", "mock-model"))
    script = [MockTurn(tool_calls=[MockToolCall("run_shell", {"command": f'{PY} -c "import time; time.sleep(60)"'})])]
    agent, _, _ = make_agent(script, workdir)
    agent.on_commit = session.append
    token = CancelToken()
    cancel_after(token, 1.5)
    agent.run_turn("slow", token)
    _, history, report = store.resume("s1")
    assert validate(history) == [] and report.interrupted_calls == 0 and report.rewritten is False
    assert history[-1].role == "assistant"


# ----------------------------------------------------- the Ctrl+C policy


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_idle_ctrl_c_leaves_the_program():
    with pytest.raises(KeyboardInterrupt):
        InterruptController().handle()


def test_first_ctrl_c_during_a_turn_only_cancels_that_turn():
    controller = InterruptController()
    token = controller.start_turn()
    controller.handle()  # must not raise
    assert token.is_cancelled
    controller.end_turn()
    next_token = controller.start_turn()
    assert not next_token.is_cancelled  # every turn gets a fresh token


def test_a_second_ctrl_c_within_the_window_quits_and_a_late_one_is_a_new_first_press():
    clock = FakeClock()
    controller = InterruptController(window=2.0, clock=clock)
    controller.start_turn()
    controller.handle()
    clock.now += 1.9
    with pytest.raises(QuitRequested):
        controller.handle()

    controller.start_turn()
    controller.handle()
    clock.now += 2.5
    controller.handle()  # too late to count as a double press: just another cancel
    clock.now += 0.5
    with pytest.raises(QuitRequested):
        controller.handle()


def test_quit_requested_is_not_swallowed_by_exception_handlers():
    assert not issubclass(QuitRequested, Exception)


def test_ctrl_c_at_a_confirmation_question_counts_as_no_and_cancels_the_turn():
    controller = InterruptController()
    token = controller.start_turn()

    def blocked_read(prompt):
        controller.handle()  # the signal arrives while input() is waiting
        raise AssertionError("the read must have been interrupted")

    with pytest.raises(KeyboardInterrupt):
        controller.guarded_input(blocked_read)("Allow? ")
    assert token.is_cancelled


def test_the_signal_handler_is_installed_and_restored():
    controller = InterruptController()
    before = signal.getsignal(signal.SIGINT)
    with controller.install():
        assert signal.getsignal(signal.SIGINT) == controller.handle
    assert signal.getsignal(signal.SIGINT) == before


# --------------------------------------------- the whole REPL, end to end


def scripted_input(*lines):
    queue = list(lines)

    def read(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    return read


def run_repl(tmp_path, workdir, script, lines, controller=None):
    ui = RecordingUI()
    mock = MockProvider(script)
    code = run(
        RunOptions(workdir=str(workdir)),
        settings=Settings(_env_file=None, mini_agent_home=tmp_path / "state"),
        provider=mock,
        ui=ui,
        input_fn=scripted_input(*lines),
        controller=controller,
    )
    return code, mock, ui


def test_first_ctrl_c_cancels_the_task_and_the_prompt_comes_back_with_the_session_intact(tmp_path, workdir):
    controller = InterruptController()
    pressed = threading.Event()

    def press_ctrl_c_once_the_turn_runs():
        wait_for(lambda: controller._token is not None, 10)
        time.sleep(0.3)
        controller.handle()
        pressed.set()

    threading.Thread(target=press_ctrl_c_once_the_turn_runs, daemon=True).start()
    script = [MockTurn(text="a long reply that never finishes", block_after_events=2), MockTurn(text="second task done")]
    code, mock, ui = run_repl(tmp_path, workdir, script, ["slow task", "second task", "/exit"], controller)
    assert pressed.is_set() and code == 0 and mock.calls == 2
    assert any("Cancelled" in m for _, m in ui.notices)
    sent = mock.requests[1].messages
    assert [m.role for m in sent] == ["user", "assistant", "user"] and sent[2].text == "second task"

    [session_id] = SessionStore(tmp_path / "state").recent_ids()
    _, history, _ = SessionStore(tmp_path / "state").resume(session_id)
    assert validate(history) == [] and history[-1].text == "second task done"


def test_the_second_quick_ctrl_c_ends_the_program_and_the_saved_session_still_resumes(tmp_path, workdir):
    script = [MockTurn(text="partial", error=QuitRequested(), error_after_events=1)]
    code, mock, ui = run_repl(tmp_path, workdir, script, ["slow task", "this line is never read"])
    assert code == 0 and mock.calls == 1
    assert any("Quitting" in m for _, m in ui.notices)

    store = SessionStore(tmp_path / "state")
    [session_id] = store.recent_ids()
    _, history, _ = store.resume(session_id)  # whatever state the quit left behind is repairable
    assert validate(history) == []


# ------------------------------------------- a real Ctrl+C on a real process


@POSIX_ONLY
def test_real_sigint_cancels_the_task_kills_the_command_tree_and_keeps_the_program_running(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "child.py").write_text("import os, time\nopen('child.pid', 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    (work / "parent.py").write_text("import subprocess, sys, time\nsubprocess.Popen([sys.executable, 'child.py'])\ntime.sleep(60)\n")
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            import sys
            from mini_agent.cli import RunOptions, run
            from mini_agent.config import Settings
            from mini_agent.providers.mock import MockProvider, MockToolCall, MockTurn
            from mini_agent.ui import ConsoleUI

            script = [
                MockTurn(tool_calls=[MockToolCall("run_shell", {{"command": "{sys.executable} parent.py"}})]),
                MockTurn(text="second task done"),
            ]
            settings = Settings(_env_file=None, mini_agent_home={str(tmp_path / "state")!r})
            code = run(RunOptions(workdir={str(work)!r}, yes=True), settings=settings, provider=MockProvider(script))
            print("EXIT", code, flush=True)
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(driver)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        proc.stdin.write("start the slow job\n")
        proc.stdin.flush()
        assert wait_for(lambda: (work / "child.pid").exists(), 30), "the command never started"
        time.sleep(0.5)
        proc.send_signal(signal.SIGINT)  # the user presses Ctrl+C once
        pid = int((work / "child.pid").read_text())
        assert wait_for(lambda: not pid_alive(pid), 15), "the grandchild survived Ctrl+C"
        assert proc.poll() is None, "the program must keep running after the first Ctrl+C"
        proc.stdin.write("second task\n/exit\n")
        proc.stdin.flush()
        output, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "EXIT 0" in output and "Cancelled" in output and "Traceback" not in output
    store = SessionStore(tmp_path / "state")
    [session_id] = store.recent_ids()
    _, history, _ = store.resume(session_id)
    assert validate(history) == [] and history[-1].text == "second task done"


@POSIX_ONLY
def test_real_double_sigint_quits_with_a_resumable_session(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    driver = tmp_path / "driver2.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            from mini_agent.cli import RunOptions, run
            from mini_agent.config import Settings
            from mini_agent.providers.mock import MockProvider, MockToolCall, MockTurn

            # The command ignores SIGTERM, so killing it takes a second (SIGTERM, wait, then SIGKILL):
            # a slow cancellation, which is the situation a second quick Ctrl+C is for.
            script = [MockTurn(tool_calls=[MockToolCall("run_shell", {{"command": "trap '' TERM; sleep 60"}})])]
            settings = Settings(_env_file=None, mini_agent_home={str(tmp_path / "state")!r})
            provider = MockProvider(script)
            print("EXIT", run(RunOptions(workdir={str(work)!r}, yes=True), settings=settings, provider=provider), flush=True)
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(driver)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        proc.stdin.write("a long task\n")
        proc.stdin.flush()
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)  # first press: the task starts to be cancelled
        time.sleep(0.3)  # signals sent back to back would be merged into one by Python
        proc.send_signal(signal.SIGINT)  # second press within two seconds, while the kill is still waiting
        output, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "EXIT 0" in output and "Quitting" in output and "Traceback" not in output
    store = SessionStore(tmp_path / "state")
    [session_id] = store.recent_ids()
    _, history, _ = store.resume(session_id)
    assert validate(history) == []
