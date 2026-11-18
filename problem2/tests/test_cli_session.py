import io

import pytest
from helpers import RecordingUI
from rich.console import Console
from typer.testing import CliRunner

from mini_agent.cli import RunOptions, app, run
from mini_agent.config import ConfigError, Settings
from mini_agent.providers.factory import build_provider
from mini_agent.providers.gemini import GeminiProvider
from mini_agent.providers.mock import MockProvider, MockToolCall, MockTurn
from mini_agent.providers.qwen import DEFAULT_BASE_URL, QwenProvider
from mini_agent.session import SessionStore
from mini_agent.ui import ConsoleUI

QWEN_KEY = "sk-test-qwen-0123456789abcdef"
GEMINI_KEY = "AIzaSyTestKey0123456789abcdefghijk"


def settings_for(tmp_path, **values):
    return Settings(_env_file=None, mini_agent_home=tmp_path / "state", **values)


def scripted_input(*lines):
    queue = list(lines)

    def read(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    return read


# ------------------------------------------------------------ settings


def test_settings_read_environment_variables_and_treat_empty_values_as_unset(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", QWEN_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.setenv("QWEN_MODEL", "qwen-max")
    settings = Settings(_env_file=None)
    assert settings.secret_values() == [QWEN_KEY]
    assert settings.gemini_model is None and settings.qwen_model == "qwen-max"


def test_settings_read_a_dotenv_file(tmp_path):
    env = tmp_path / "test.env"
    env.write_text(f"GEMINI_API_KEY={GEMINI_KEY}\nGEMINI_MODEL=gemini-x\nUNRELATED=1\n", encoding="utf-8")
    settings = Settings(_env_file=env)
    assert settings.gemini_model == "gemini-x" and settings.secret_values() == [GEMINI_KEY]


def test_api_keys_never_show_up_in_repr_or_str():
    settings = Settings(_env_file=None, dashscope_api_key=QWEN_KEY, gemini_api_key=GEMINI_KEY)
    shown = repr(settings) + str(settings) + repr(settings.model_dump())
    assert QWEN_KEY not in shown and GEMINI_KEY not in shown


def test_state_dir_defaults_to_the_home_directory_but_can_be_moved(tmp_path):
    assert Settings(_env_file=None).state_dir.name == ".mini-agent"
    assert Settings(_env_file=None, mini_agent_home=tmp_path).state_dir == tmp_path


# ------------------------------------------------------------- factory


def test_qwen_is_the_default_provider_and_uses_the_default_endpoint(tmp_path):
    provider = build_provider(None, None, settings_for(tmp_path, dashscope_api_key=QWEN_KEY))
    assert isinstance(provider, QwenProvider) and provider.model == "qwen-plus"
    assert str(provider._client.base_url).rstrip("/") == DEFAULT_BASE_URL


def test_qwen_model_and_base_url_can_be_overridden(tmp_path):
    settings = settings_for(tmp_path, dashscope_api_key=QWEN_KEY, qwen_base_url="https://example.test/v1")
    provider = build_provider("QWEN", "qwen-max", settings)
    assert provider.model == "qwen-max" and str(provider._client.base_url).startswith("https://example.test/v1")


def test_gemini_needs_a_model_and_a_key(tmp_path):
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_provider("gemini", "m", settings_for(tmp_path))
    with pytest.raises(ConfigError, match="--model or set GEMINI_MODEL"):
        build_provider("gemini", None, settings_for(tmp_path, gemini_api_key=GEMINI_KEY))
    provider = build_provider("gemini", "gemini-x", settings_for(tmp_path, gemini_api_key=GEMINI_KEY))
    assert isinstance(provider, GeminiProvider) and provider.model == "gemini-x"
    from_env = build_provider(None, None, settings_for(tmp_path, gemini_api_key=GEMINI_KEY, gemini_model="g", mini_agent_provider="gemini"))
    assert isinstance(from_env, GeminiProvider)


def test_missing_qwen_key_and_unknown_provider_have_clear_messages(tmp_path):
    with pytest.raises(ConfigError, match="DASHSCOPE_API_KEY is not set"):
        build_provider("qwen", None, settings_for(tmp_path))
    with pytest.raises(ConfigError, match="Unknown provider 'openai'.*qwen, gemini"):
        build_provider("openai", None, settings_for(tmp_path))


# ---------------------------------------------------------------- REPL via run()


def run_cli(tmp_path, workdir, script, lines, *, options=None, ui=None, provider="default"):
    ui = ui or RecordingUI(default=True)
    mock = MockProvider(script) if provider == "default" else provider
    code = run(
        options or RunOptions(workdir=str(workdir)),
        settings=settings_for(tmp_path),
        provider=mock,
        ui=ui,
        input_fn=scripted_input(*lines),
    )
    return code, mock, ui


def notices(ui):
    return "\n".join(message for _, message in ui.notices)


def test_two_tasks_in_one_session_share_context_and_the_session_is_saved(tmp_path, workdir):
    code, mock, ui = run_cli(tmp_path, workdir, [MockTurn(text="one done"), MockTurn(text="two done")], ["first task", "second task", "/exit"])
    assert code == 0 and mock.calls == 2
    assert [m.text for m in mock.requests[1].messages if m.role == "user"] == ["first task", "second task"]
    store = SessionStore(tmp_path / "state")
    [session_id] = store.recent_ids()
    _, history, _ = store.resume(session_id)
    assert [m.text for m in history] == ["first task", "one done", "second task", "two done"]
    assert f"--resume {session_id}" in notices(ui)


def test_help_unknown_commands_and_blank_lines_never_reach_the_model(tmp_path, workdir):
    code, mock, ui = run_cli(tmp_path, workdir, [], ["", "   ", "/help", "/bogus", "/exit"])
    assert code == 0 and mock.calls == 0
    text = notices(ui)
    assert "/exit" in text and "Unknown command /bogus" in text


def test_a_path_like_message_is_still_a_task(tmp_path, workdir):
    code, mock, _ = run_cli(tmp_path, workdir, [MockTurn(text="ok")], ["/etc/hosts explain this file", "/exit"])
    assert mock.calls == 1 and mock.requests[0].messages[0].text == "/etc/hosts explain this file"


def test_end_of_input_and_ctrl_c_at_the_prompt_leave_cleanly(tmp_path, workdir):
    assert run_cli(tmp_path, workdir, [], [])[0] == 0  # EOF straight away

    def interrupting(prompt):
        raise KeyboardInterrupt

    ui = RecordingUI()
    assert run(RunOptions(workdir=str(workdir)), settings=settings_for(tmp_path), provider=MockProvider([]), ui=ui, input_fn=interrupting) == 0


def test_ctrl_c_during_a_turn_returns_to_the_prompt_and_the_conversation_continues(tmp_path, workdir):
    script = [MockTurn(error=KeyboardInterrupt()), MockTurn(text="back")]
    code, mock, ui = run_cli(tmp_path, workdir, script, ["slow task", "next task", "/exit"])
    assert code == 0 and mock.calls == 2
    assert "Interrupted" in notices(ui)
    # the strict mock accepted the second request, so the history was still valid after the interruption
    assert mock.requests[1].messages[-1].text == "next task"


def test_yes_flag_skips_confirmation_and_otherwise_the_user_is_asked(tmp_path, workdir):
    script = [MockTurn(tool_calls=[MockToolCall("write_file", {"path": "a.txt", "content": "x"})]), MockTurn(text="done")]
    ui = RecordingUI(default=False)
    run_cli(tmp_path, workdir, script, ["write it", "/exit"], ui=ui)
    assert len(ui.confirms) == 1 and not (workdir / "a.txt").exists()

    script = [MockTurn(tool_calls=[MockToolCall("write_file", {"path": "a.txt", "content": "x"})]), MockTurn(text="done")]
    ui = RecordingUI(default=False)
    run_cli(tmp_path, workdir, script, ["write it", "/exit"], options=RunOptions(workdir=str(workdir), yes=True), ui=ui)
    assert ui.confirms == [] and (workdir / "a.txt").read_text() == "x"


def test_max_steps_option_is_applied(tmp_path, workdir):
    script = [MockTurn(tool_calls=[MockToolCall("list_files")]) for _ in range(5)]
    code, mock, ui = run_cli(tmp_path, workdir, script, ["loop", "/exit"], options=RunOptions(workdir=str(workdir), max_steps=2))
    assert mock.calls == 2 and "maximum of 2 steps" in notices(ui)


def test_log_file_option_and_default_log_location(tmp_path, workdir):
    custom = tmp_path / "custom" / "my.log"
    run_cli(tmp_path, workdir, [MockTurn(text="a")], ["hello", "/exit"], options=RunOptions(workdir=str(workdir), log_file=str(custom)))
    assert custom.is_file() and '"event": "request"' in custom.read_text(encoding="utf-8")

    run_cli(tmp_path, workdir, [MockTurn(text="a")], ["hello", "/exit"])
    assert list((tmp_path / "state" / "logs").glob("*.log"))


def test_the_api_key_from_settings_is_redacted_in_the_log(tmp_path, workdir):
    settings = settings_for(tmp_path, dashscope_api_key=QWEN_KEY)
    log = tmp_path / "k.log"
    run(
        RunOptions(workdir=str(workdir), log_file=str(log)),
        settings=settings,
        provider=MockProvider([MockTurn(text="ok")]),
        ui=RecordingUI(),
        input_fn=scripted_input(f"my key is {QWEN_KEY}", "/exit"),
    )
    assert QWEN_KEY not in log.read_text(encoding="utf-8")


# ------------------------------------------------------------------ resume


def first_session_id(tmp_path):
    return SessionStore(tmp_path / "state").recent_ids()[0]


def test_resume_restores_the_history_and_continues_the_conversation(tmp_path, workdir):
    run_cli(tmp_path, workdir, [MockTurn(text="first reply")], ["first task", "/exit"])
    session_id = first_session_id(tmp_path)

    code, mock, ui = run_cli(
        tmp_path, workdir, [MockTurn(text="second reply")], ["second task", "/exit"], options=RunOptions(resume=session_id)
    )
    assert code == 0
    sent = mock.requests[0].messages
    assert [m.text for m in sent] == ["first task", "first reply", "second task"]
    assert "resumed with 2 messages" in notices(ui)
    _, history, _ = SessionStore(tmp_path / "state").resume(session_id)
    assert [m.text for m in history][-1] == "second reply"


def test_resume_after_a_crash_in_the_middle_of_tool_execution_tells_the_user_and_the_model(tmp_path, workdir):
    run_cli(tmp_path, workdir, [MockTurn(text="ok")], ["task", "/exit"])
    session_id = first_session_id(tmp_path)
    store = SessionStore(tmp_path / "state")
    session, _, _ = store.resume(session_id)
    from mini_agent.messages import Message, ToolUseBlock

    session.append(Message.user("run the tests"))
    session.append(Message(role="assistant", content=[ToolUseBlock(id="c1", name="run_shell", input={"command": "pytest"})]))
    with open(session.path, "ab") as f:
        f.write(b'{"type": "message", "mess')

    code, mock, ui = run_cli(tmp_path, workdir, [MockTurn(text="checked")], ["what is the state?", "/exit"], options=RunOptions(resume=session_id))
    assert code == 0
    text = notices(ui)
    assert "incomplete" in text and "1 tool call(s)" in text
    sent = mock.requests[0].messages
    assert sent[-3].tool_results[0].content.startswith("The program exited while this tool call was running")


def test_resume_without_workdir_uses_the_saved_one_and_a_different_workdir_is_refused(tmp_path, workdir):
    run_cli(tmp_path, workdir, [MockTurn(text="ok")], ["task", "/exit"])
    session_id = first_session_id(tmp_path)
    other = tmp_path / "other"
    other.mkdir()

    ui = RecordingUI()
    code = run(RunOptions(resume=session_id, workdir=str(other)), settings=settings_for(tmp_path), provider=MockProvider([]), ui=ui, input_fn=scripted_input("/exit"))
    assert code == 2 and "belongs to the working directory" in notices(ui)

    code, _, ui = run_cli(tmp_path, workdir, [], ["/exit"], options=RunOptions(resume=session_id))
    assert code == 0 and str(workdir.resolve()) in notices(ui)
    code, _, _ = run_cli(tmp_path, workdir, [], ["/exit"], options=RunOptions(resume=session_id, workdir=str(workdir)))
    assert code == 0


def test_resume_of_an_unknown_session_is_a_clean_error(tmp_path, workdir):
    code, _, ui = run_cli(tmp_path, workdir, [], [], options=RunOptions(resume="20990101-000000-abcdef"))
    assert code == 2 and "was not found" in notices(ui) and ui.notices[-1][0] == "error"


# ------------------------------------------------------ errors and the real CLI


def test_a_missing_working_directory_and_a_missing_key_are_clean_errors(tmp_path):
    ui = RecordingUI()
    code = run(RunOptions(workdir=str(tmp_path / "missing")), settings=settings_for(tmp_path), provider=MockProvider([]), ui=ui, input_fn=scripted_input())
    assert code == 2 and "does not exist" in notices(ui)

    (tmp_path / "w").mkdir()
    ui = RecordingUI()
    code = run(RunOptions(workdir=str(tmp_path / "w")), settings=settings_for(tmp_path), ui=ui, input_fn=scripted_input())
    assert code == 2 and "DASHSCOPE_API_KEY" in notices(ui)
    assert not (tmp_path / "state" / "sessions").exists()  # no session file is created when startup fails


def test_a_fatal_provider_error_does_not_end_the_session(tmp_path, workdir):
    from mini_agent.providers.errors import FatalError

    script = [MockTurn(error=FatalError("Authentication failed (401): check API key", 401)), MockTurn(text="fixed")]
    code, mock, ui = run_cli(tmp_path, workdir, script, ["one", "two", "/exit"])
    assert code == 0 and mock.calls == 2 and "401" in notices(ui)


def test_the_command_line_shows_all_required_options():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for option in ("--workdir", "--provider", "--model", "--yes", "--resume", "--max-steps", "--log-file"):
        assert option in result.output


def test_the_command_line_reports_a_missing_key_without_a_traceback(tmp_path):
    result = CliRunner().invoke(app, ["--workdir", str(tmp_path)], env={"MINI_AGENT_HOME": str(tmp_path / "state")})
    assert result.exit_code == 2
    assert "DASHSCOPE_API_KEY is not set" in result.output and "Traceback" not in result.output


def test_console_ui_error_notices_are_visible_on_a_real_console():
    out = io.StringIO()
    ConsoleUI(Console(file=out)).notice("Error: boom", "error")
    assert "Error: boom" in out.getvalue()
