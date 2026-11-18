import subprocess
import sys
import threading
import time

import pytest

from mini_agent.agent.cancel import CancelToken
from mini_agent.tools.shell import IS_WINDOWS, run_command, sanitized_env

PY = f'"{sys.executable}"'


def pid_alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_until(predicate, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_returns_exit_code_stdout_and_stderr(run_tool):
    cmd = f"{PY} -c \"import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)\""
    result = run_tool("run_shell", command=cmd)
    assert "exit_code: 3" in result.content
    assert "--- stdout ---\nout" in result.content and "--- stderr ---\nerr" in result.content
    assert not result.is_error  # a failing command is information, not a tool failure


def test_runs_in_the_working_directory(run_tool, workdir):
    (workdir / "marker.txt").write_text("here")
    result = run_tool("run_shell", command=f"{PY} -c \"import os; print(os.listdir('.'))\"")
    assert "marker.txt" in result.content


def test_stdin_is_closed_so_interactive_commands_do_not_hang(run_tool):
    result = run_tool("run_shell", command=f"{PY} -c \"import sys; print(repr(sys.stdin.read()))\"", timeout=20)
    assert "''" in result.content and not result.is_error


def test_api_keys_are_not_visible_to_commands(run_tool, monkeypatch):
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-super-secret")
    monkeypatch.setenv("VISIBLE_SETTING", "ok")
    result = run_tool(
        "run_shell",
        command=f"{PY} -c \"import os; print(os.environ.get('FAKE_PROVIDER_API_KEY'), os.environ.get('VISIBLE_SETTING'))\"",
    )
    assert "sk-super-secret" not in result.content
    assert "None ok" in result.content


def test_sanitized_env_drops_only_api_keys():
    env = sanitized_env({"OPENAI_API_KEY": "a", "gemini_api_key": "b", "PATH": "p", "MY_KEYBOARD": "k"})
    assert env == {"PATH": "p", "MY_KEYBOARD": "k"}


def test_long_output_keeps_head_and_tail(run_tool):
    code = "for i in range(20000): print('row', i)"
    result = run_tool("run_shell", command=f'{PY} -c "{code}"', timeout=60)
    assert "row 0" in result.content and "row 19999" in result.content
    assert "bytes omitted" in result.content and len(result.content) < 20_000


def test_timeout_kills_the_command_and_reports_it(run_tool):
    result = run_tool("run_shell", command=f'{PY} -c "import time; time.sleep(60)"', timeout=1)
    assert result.is_error and "Timed out after 1s" in result.content


def test_default_timeout_comes_from_the_context(ctx, run_tool):
    ctx.shell_timeout = 1
    result = run_tool("run_shell", command=f'{PY} -c "import time; time.sleep(60)"')
    assert result.is_error and "Timed out after 1s" in result.content


def test_timeout_argument_is_validated(run_tool):
    assert run_tool("run_shell", command="echo hi", timeout=0).is_error
    assert run_tool("run_shell", command="echo hi", timeout=100000).is_error
    assert run_tool("run_shell", command="").is_error


def test_timeout_kills_grandchild_processes_too(workdir):
    (workdir / "child.py").write_text("import os, time\nopen('child.pid', 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    (workdir / "parent.py").write_text(
        "import subprocess, sys, time\nsubprocess.Popen([sys.executable, 'child.py'])\ntime.sleep(60)\n"
    )
    outcome = run_command(f"{PY} parent.py", str(workdir), timeout=4, cancel=CancelToken())
    assert outcome.timed_out
    pid_file = workdir / "child.pid"
    assert pid_file.exists(), "grandchild never started; the test machine is too slow"
    pid = int(pid_file.read_text())
    assert wait_until(lambda: not pid_alive(pid), 5), "grandchild process survived the timeout"


def test_cancel_stops_a_running_command_quickly(workdir):
    cancel = CancelToken()
    threading.Timer(0.5, cancel.cancel).start()
    started = time.monotonic()
    outcome = run_command(f'{PY} -c "import time; time.sleep(60)"', str(workdir), timeout=60, cancel=cancel)
    assert outcome.cancelled and time.monotonic() - started < 10


def test_output_before_cancellation_is_kept(workdir):
    cancel = CancelToken()
    code = "import sys, time; print('partial output', flush=True); time.sleep(60)"
    threading.Timer(2.0, cancel.cancel).start()
    outcome = run_command(f'{PY} -c "{code}"', str(workdir), timeout=60, cancel=cancel)
    assert outcome.cancelled and "partial output" in outcome.stdout


def test_background_child_holding_the_pipes_does_not_hang_the_tool(workdir):
    if IS_WINDOWS:
        command = f'start /B "" {PY} -c "import time; time.sleep(15)"'
    else:
        command = "sleep 15 &"
    started = time.monotonic()
    outcome = run_command(command, str(workdir), timeout=60, cancel=CancelToken())
    assert time.monotonic() - started < 10
    assert not outcome.timed_out and not outcome.cancelled
