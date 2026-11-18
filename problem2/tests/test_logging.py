import io
import json

from helpers import make_agent
from rich.console import Console

from mini_agent.logging_ import REDACTED, Redactor, RequestLogger
from mini_agent.messages import Usage
from mini_agent.providers.errors import FatalError
from mini_agent.providers.mock import MockToolCall, MockTurn
from mini_agent.tokens import estimate_message_tokens, estimate_request_tokens, estimate_text_tokens
from mini_agent.messages import Message, ToolResultBlock, ToolUseBlock
from mini_agent.ui import ConsoleUI, summarize_args, summarize_result

FAKE_KEY = "sk-test-FAKEKEY1234567890abcdef"


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------- redaction


def test_exact_secret_values_and_common_key_patterns_are_removed():
    redactor = Redactor(["my-very-secret-value"])
    text = "key=my-very-secret-value other sk-abcdefghijklmnop1234 AIzaSyA1234567890abcdefghij Authorization: Bearer abcdefghijklmnop12345"
    out = redactor.redact(text)
    for leaked in ("my-very-secret-value", "sk-abcdefghijklmnop1234", "AIzaSyA1234567890abcdefghij", "abcdefghijklmnop12345"):
        assert leaked not in out
    assert out.count(REDACTED) == 4 and "Bearer" in out


def test_json_escaped_secret_is_also_removed():
    secret = 'pa"ss\\word-123456'
    line = json.dumps({"text": secret})
    assert secret not in Redactor([secret]).redact(line)
    assert "pa" not in Redactor([secret]).redact(line).replace("[REDACTED]", "")


def test_very_short_values_are_not_treated_as_secrets():
    assert Redactor(["abc"]).redact("abc abc") == "abc abc"


# ----------------------------------------------------------- log content


def test_a_full_turn_is_logged_and_the_key_never_appears(workdir):
    log_path = workdir.parent / "logs" / "agent.log"
    redactor = Redactor([FAKE_KEY])
    logger = RequestLogger(log_path, redactor)
    (workdir / "config.txt").write_text(f"api_key={FAKE_KEY}\n")
    script = [
        MockTurn(text="reading", tool_calls=[MockToolCall("read_file", {"path": "config.txt"})]),
        MockTurn(error=FatalError(f"Authentication failed (401) for key {FAKE_KEY}", 401)),
    ]
    agent, provider, ui = make_agent(script, workdir, logger=logger, redactor=redactor)
    result = agent.run_turn(f"my key is {FAKE_KEY}, please read config.txt")
    assert result.status == "error"

    raw = log_path.read_text(encoding="utf-8")
    assert FAKE_KEY not in raw and "FAKEKEY1234567890" not in raw
    assert REDACTED in raw
    assert "Authorization" not in raw
    assert FAKE_KEY not in result.error
    assert all(FAKE_KEY not in message for _, message in ui.notices)


def test_control_without_the_secret_registered_the_log_does_contain_user_text(workdir):
    # Proves the "key never appears" assertion above is meaningful: the log really records
    # user input, so a value only disappears when the redactor knows about it.
    plain_secret = "plain-secret-value-987654"
    log_path = workdir.parent / "control.log"
    agent, _, _ = make_agent([MockTurn(text="ok")], workdir, logger=RequestLogger(log_path, Redactor()))
    agent.run_turn(f"the value is {plain_secret}")
    assert plain_secret in log_path.read_text(encoding="utf-8")


def test_log_has_complete_request_and_response_records(workdir):
    log_path = workdir.parent / "agent.log"
    logger = RequestLogger(log_path)
    (workdir / "a.txt").write_text("hello\n")
    script = [MockTurn(tool_calls=[MockToolCall("read_file", {"path": "a.txt"})]), MockTurn(text="done")]
    agent, _, _ = make_agent(script, workdir, logger=logger)
    agent.run_turn("read a.txt")

    records = read_log(log_path)
    assert [r["event"] for r in records] == ["request", "response", "tool", "request", "response", "turn_end"]
    first, second = records[0], records[3]
    assert first["messages"][0]["content"][0]["text"] == "read a.txt"
    assert first["system"] and "read_file" in first["tools"]
    assert len(second["messages"]) == 3  # user, assistant(tool_use), user(tool_result)
    assert records[1]["message"]["content"][0]["type"] == "tool_use" and records[1]["usage"]["input_tokens"] == 10
    assert records[2]["name"] == "read_file" and "hello" in records[2]["result"]
    assert records[-1]["status"] == "completed" and records[-1]["usage"]["output_tokens"] == 10
    assert all("ts" in r for r in records)


def test_disabled_logger_writes_nothing_and_a_broken_path_does_not_crash(workdir, tmp_path):
    RequestLogger(None).log("x", a=1)
    blocked = tmp_path / "file.txt"
    blocked.write_text("x")
    logger = RequestLogger(tmp_path / "ok.log")
    logger.path = blocked / "cannot" / "log.txt"  # parent is a file, so opening fails
    logger.log("event")
    assert logger.write_errors == 1


# ------------------------------------------------------------------ UI


def make_ui(answers=(), interactive=True):
    out = io.StringIO()
    answers = list(answers)
    ui = ConsoleUI(Console(file=out, width=100, force_terminal=False), input_fn=lambda prompt: answers.pop(0), interactive=interactive)
    return ui, out


def test_confirmation_accepts_only_explicit_yes():
    for answer, expected in [("y", True), ("YES", True), (" y ", True), ("n", False), ("", False), ("maybe", False)]:
        ui, out = make_ui([answer])
        assert ui.confirm("run_shell", "$ ls") is expected
        assert "$ ls" in out.getvalue()


def test_confirmation_on_closed_stdin_or_non_interactive_terminal_is_a_rejection():
    def eof(prompt):
        raise EOFError

    out = io.StringIO()
    assert ConsoleUI(Console(file=out), input_fn=eof, interactive=True).confirm("run_shell", "$ ls") is False
    ui, out = make_ui(["y"], interactive=False)
    assert ui.confirm("run_shell", "$ ls") is False and "--yes" in out.getvalue()


def test_model_text_tool_calls_results_and_usage_render_distinctly():
    ui, out = make_ui()
    ui.text_delta("Hello ")
    ui.text_delta("world")
    ui.tool_call("read_file", {"path": "a.py"})
    ui.tool_result("read_file", "line1\nline2\nline3", False)
    ui.tool_result("read_file", "boom", True)
    ui.usage(Usage(input_tokens=12, output_tokens=3, estimated=True))
    lines = out.getvalue().splitlines()
    assert lines[0] == "Hello world"
    assert lines[1] == "> read_file path=a.py"
    assert lines[2] == "  ok: line1  (+2 more lines)"
    assert lines[3] == "  error: boom"
    assert lines[4] == "tokens: input 12 / output 3 (estimated)"


def test_markup_like_text_is_printed_literally():
    ui, out = make_ui()
    ui.text_delta("[bold]not markup[/bold] [red]")
    ui.tool_call("run_shell", {"command": "echo [x]"})
    assert "[bold]not markup[/bold] [red]" in out.getvalue() and "echo [x]" in out.getvalue()


def test_summaries_are_short_and_hide_file_contents():
    assert summarize_args({"path": "a.py", "content": "x" * 5000}) == "path=a.py content=<5000 chars>"
    assert len(summarize_args({"command": "y" * 500})) < 120
    assert summarize_args({}) == ""
    assert summarize_result("") == "(no output)"
    assert summarize_result("a\n\nb") == "a  (+1 more lines)"


# ---------------------------------------------------------------- tokens


def test_token_estimates_scale_with_length_and_weigh_cjk_higher():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("a" * 400) == 100
    assert estimate_text_tokens("你" * 100) == 150
    message = Message(role="assistant", content=[ToolUseBlock(id="a", name="read_file", input={"path": "x" * 40})])
    result = Message(role="user", content=[ToolResultBlock(tool_use_id="a", content="y" * 4000)])
    assert estimate_message_tokens(result) > 1000 > estimate_message_tokens(message)
    assert estimate_request_tokens("sys", [result]) > estimate_message_tokens(result)
