import json
import os

import pytest
from helpers import RecordingUI, make_agent

from mini_agent.history import validate
from mini_agent.messages import Message, ToolResultBlock, ToolUseBlock
from mini_agent.providers.mock import MockProvider, MockToolCall, MockTurn
from mini_agent.session import (
    RESUME_MISSING_RESULT,
    SessionError,
    SessionStore,
    new_meta,
)


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "state")


def create(store, session_id="s1", workdir="/work"):
    return store.create(new_meta(session_id, workdir, "mock", "mock-model"))


def tool_use_message(*ids):
    return Message(role="assistant", content=[ToolUseBlock(id=i, name="read_file", input={"path": "a"}) for i in ids])


def lines_of(session):
    return session.path.read_bytes().split(b"\n")


# ------------------------------------------------------------ writing


def test_each_message_is_one_json_line_after_the_meta_header(store):
    session = create(store)
    session.append(Message.user("hello 你好"))
    session.append(Message.assistant("hi"))
    lines = session.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["type"] for l in lines] == ["meta", "message", "message"]
    assert json.loads(lines[0])["session_id"] == "s1"
    assert "你好" in lines[1]  # not \u-escaped


def test_session_files_live_under_the_state_dir_and_ids_are_unique(store):
    ids = {store.new_id() for _ in range(20)}
    assert len(ids) == 20
    session = create(store)
    assert session.path.parent == store.dir and store.dir.name == "sessions"


def test_creating_the_same_session_twice_is_an_error(store):
    create(store)
    with pytest.raises(FileExistsError):
        create(store)


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a b", "", ".hidden", "x" * 100])
def test_session_ids_cannot_escape_the_sessions_directory(store, bad):
    with pytest.raises(SessionError, match="Invalid session id"):
        store.resume(bad)


# ----------------------------------------------------------- resuming


def test_a_clean_session_round_trips_unchanged_and_is_not_rewritten(store):
    session = create(store)
    messages = [
        Message.user("fix it"),
        tool_use_message("a"),
        Message(role="user", content=[ToolResultBlock(tool_use_id="a", content="data")]),
        Message.assistant("done"),
    ]
    for m in messages:
        session.append(m)
    before = session.path.read_bytes()
    resumed, history, report = store.resume("s1")
    assert history == messages and resumed.meta.workdir == "/work"
    assert report.rewritten is False and report.interrupted_calls == 0 and report.dropped_partial_line is False
    assert session.path.read_bytes() == before


def test_crash_during_a_tool_leaves_a_dangling_call_that_is_repaired_with_an_explicit_note(store):
    session = create(store)
    session.append(Message.user("do it"))
    session.append(tool_use_message("a", "b"))  # written BEFORE the tools ran; the program died after this
    resumed, history, report = store.resume("s1")
    assert validate(history) == []
    assert report.interrupted_calls == 2 and report.rewritten is True
    results = history[2].tool_results
    assert [r.tool_use_id for r in results] == ["a", "b"] and all(r.is_error for r in results)
    assert "side effects are unknown" in results[0].content and results[0].content == RESUME_MISSING_RESULT


def test_half_written_last_line_is_dropped_and_the_file_is_cleaned_so_appends_stay_valid(store):
    session = create(store)
    session.append(Message.user("one"))
    session.append(Message.assistant("two"))
    with open(session.path, "ab") as f:
        f.write(b'{"type": "message", "message": {"role": "user", "content": [{"type": "te')
    resumed, history, report = store.resume("s1")
    assert [m.text for m in history] == ["one", "two"]
    assert report.dropped_partial_line is True and report.rewritten is True
    assert resumed.path.read_bytes().endswith(b"\n")

    resumed.append(Message.user("three"))  # must not be glued to the broken fragment
    _, history, report = store.resume("s1")
    assert [m.text for m in history] == ["one", "two", "three"] and report.dropped_partial_line is False


def test_a_complete_last_line_that_only_lacks_its_newline_is_kept(store):
    session = create(store)
    session.append(Message.user("one"))
    session.path.write_bytes(session.path.read_bytes().rstrip(b"\n"))
    resumed, history, report = store.resume("s1")
    assert [m.text for m in history] == ["one"] and report.dropped_partial_line is False and report.rewritten is True
    assert resumed.path.read_bytes().endswith(b"\n")


def test_a_cut_in_the_middle_of_a_multibyte_character_is_handled(store):
    session = create(store)
    session.append(Message.user("one"))
    with open(session.path, "ab") as f:
        f.write('{"type": "message", "message": {"text": "你'.encode("utf-8")[:-1])
    _, history, report = store.resume("s1")
    assert [m.text for m in history] == ["one"] and report.dropped_partial_line is True


def test_crash_after_the_user_message_only_resumes_with_a_valid_history(store):
    session = create(store)
    session.append(Message.user("first task"))
    _, history, _ = store.resume("s1")
    assert [m.role for m in history] == ["user"] and validate(history) == []


def test_corruption_in_the_middle_is_reported_not_silently_skipped(store):
    session = create(store)
    session.append(Message.user("one"))
    session.append(Message.assistant("two"))
    lines = session.path.read_bytes().split(b"\n")
    lines[1] = b"this is not json"
    session.path.write_bytes(b"\n".join(lines))
    with pytest.raises(SessionError, match="corrupted at line 2"):
        store.resume("s1")


def test_a_file_without_a_valid_header_cannot_be_resumed(store):
    store.dir.mkdir(parents=True)
    (store.dir / "bad.jsonl").write_bytes(b'{"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "x"}]}}\n')
    with pytest.raises(SessionError, match="no valid header"):
        store.resume("bad")
    (store.dir / "empty.jsonl").write_bytes(b'{"type": "meta"')  # crashed while creating the file
    with pytest.raises(SessionError, match="no valid header"):
        store.resume("empty")


def test_unknown_session_lists_recent_ones(store):
    with pytest.raises(SessionError, match="no saved sessions"):
        store.resume("nope")
    create(store, "alpha")
    create(store, "beta")
    with pytest.raises(SessionError) as info:
        store.resume("nope")
    assert "alpha" in str(info.value) and "beta" in str(info.value)


def test_orphan_tool_result_on_disk_is_dropped_on_resume(store):
    session = create(store)
    session.append(Message.user("hi"))
    session.append(Message.assistant("hello"))
    session.append(Message(role="user", content=[ToolResultBlock(tool_use_id="ghost", content="x")]))
    _, history, report = store.resume("s1")
    assert validate(history) == [] and report.rewritten is True


# ------------- crash recovery end to end, with the real agent and mock provider


class Crash(BaseException):
    """Stands in for the process being killed: nothing after this point runs, nothing more is saved."""


def test_agent_killed_mid_tools_resumes_into_a_valid_conversation(store, workdir):
    session = create(store, workdir=str(workdir))

    def on_commit(message):
        if message.role == "user" and message.tool_results:
            raise Crash  # the program dies before the tool results are saved
        session.append(message)

    script = [MockTurn(tool_calls=[MockToolCall("write_file", {"path": "a.txt", "content": "x"}), MockToolCall("list_files")])]
    agent, _, _ = make_agent(script, workdir, ui=RecordingUI(default=True))
    agent.on_commit = on_commit
    with pytest.raises(Crash):
        agent.run_turn("create a.txt and list files")
    saved_roles = [json.loads(l)["message"]["role"] for l in session.path.read_text(encoding="utf-8").splitlines()[1:]]
    assert saved_roles == ["user", "assistant"]  # the tool_use was saved before the tools ran

    resumed, history, report = store.resume("s1")
    assert report.interrupted_calls == 2
    follow_up = MockProvider([MockTurn(text="I checked and continued.")])
    agent2, _, _ = make_agent([], workdir)
    agent2.provider = follow_up  # strict mock: it answers 400 if the history is not valid
    agent2.history = history
    agent2.on_commit = resumed.append
    assert agent2.run_turn("what happened?").status == "completed"
    assert follow_up.calls == 1 and validate(agent2.history) == []

    _, again, _ = store.resume("s1")  # the persisted file is valid too, after the new turn
    assert validate(again) == [] and again[-1].text == "I checked and continued."
