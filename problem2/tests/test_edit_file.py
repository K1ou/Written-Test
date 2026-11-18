import pytest


def write(workdir, name, text):
    path = workdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def edit(run_tool, **args):
    return run_tool("edit_file", **args)


def test_single_match_is_replaced(run_tool, workdir):
    path = write(workdir, "a.py", "x = 1\ny = 2\n")
    result = edit(run_tool, path="a.py", old_string="y = 2", new_string="y = 3")
    assert not result.is_error and "line 2" in result.content
    assert path.read_text() == "x = 1\ny = 3\n"


def test_no_match_returns_error_with_reason(run_tool, workdir):
    path = write(workdir, "a.py", "x = 1\n")
    result = edit(run_tool, path="a.py", old_string="z = 9", new_string="z = 10")
    assert result.is_error and "not found" in result.content and "read_file" in result.content
    assert path.read_text() == "x = 1\n"


def test_multiple_matches_returns_error_listing_lines(run_tool, workdir):
    path = write(workdir, "a.py", "print(1)\nprint(1)\nprint(1)\n")
    result = edit(run_tool, path="a.py", old_string="print(1)", new_string="print(2)")
    assert result.is_error and "3 times" in result.content and "1, 2, 3" in result.content
    assert "context" in result.content
    assert path.read_text() == "print(1)\nprint(1)\nprint(1)\n"


def test_overlapping_matches_count_as_ambiguous(run_tool, workdir):
    write(workdir, "a.txt", "aaa")
    result = edit(run_tool, path="a.txt", old_string="aa", new_string="b")
    assert result.is_error and "2 times" in result.content


def test_more_context_makes_an_ambiguous_edit_unique(run_tool, workdir):
    path = write(workdir, "a.py", "print(1)\nprint(1)\n")
    result = edit(run_tool, path="a.py", old_string="print(1)\nprint(1)", new_string="print(2)")
    assert not result.is_error
    assert path.read_text() == "print(2)\n"


def test_whitespace_mismatch_is_explained(run_tool, workdir):
    write(workdir, "a.py", "def f():\n    return 1\n")
    result = edit(run_tool, path="a.py", old_string="return 1", new_string="return 2\n")
    assert not result.is_error  # substring match works
    result = edit(run_tool, path="a.py", old_string="  return 2 ", new_string="return 3")
    assert result.is_error and "whitespace" in result.content


def test_identical_strings_are_rejected(run_tool, workdir):
    write(workdir, "a.txt", "hello")
    result = edit(run_tool, path="a.txt", old_string="hello", new_string="hello")
    assert result.is_error and "identical" in result.content


def test_empty_old_string_is_rejected(run_tool, workdir):
    path = write(workdir, "a.txt", "hello")
    result = edit(run_tool, path="a.txt", old_string="", new_string="x")
    assert result.is_error and "empty" in result.content
    assert path.read_text() == "hello"


def test_missing_file_and_directory_targets(run_tool, workdir):
    (workdir / "d").mkdir()
    missing = edit(run_tool, path="nope.txt", old_string="a", new_string="b")
    assert missing.is_error and "not found" in missing.content.lower()
    directory = edit(run_tool, path="d", old_string="a", new_string="b")
    assert directory.is_error and "directory" in directory.content


def test_multiline_replacement_and_deletion(run_tool, workdir):
    path = write(workdir, "a.py", "a\nb\nc\nd\n")
    assert not edit(run_tool, path="a.py", old_string="b\nc\n", new_string="").is_error
    assert path.read_text() == "a\nd\n"


def test_unicode_content_is_preserved(run_tool, workdir):
    path = write(workdir, "a.txt", "你好，世界 🌍\n第二行\n")
    result = edit(run_tool, path="a.txt", old_string="世界", new_string="Agent")
    assert not result.is_error
    assert path.read_text(encoding="utf-8") == "你好，Agent 🌍\n第二行\n"


def test_edit_does_not_touch_untouched_line_endings_or_bom(run_tool, workdir):
    path = workdir / "a.txt"
    path.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\nthree\r\n")
    result = edit(run_tool, path="a.txt", old_string="two", new_string="TWO")
    assert not result.is_error
    assert path.read_bytes() == b"\xef\xbb\xbfone\r\nTWO\r\nthree\r\n"


def test_crlf_file_with_lf_old_string_explains_the_cause(run_tool, workdir):
    path = workdir / "a.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    result = edit(run_tool, path="a.txt", old_string="one\ntwo", new_string="x")
    assert result.is_error and "CRLF" in result.content
    assert path.read_bytes() == b"one\r\ntwo\r\n"


def test_non_utf8_file_is_refused_not_corrupted(run_tool, workdir):
    path = workdir / "gbk.txt"
    data = "你好".encode("gbk")
    path.write_bytes(data)
    result = edit(run_tool, path="gbk.txt", old_string="a", new_string="b")
    assert result.is_error and "UTF-8" in result.content
    assert path.read_bytes() == data


def test_preview_shows_diff_and_raises_the_same_errors(workdir, ctx):
    from mini_agent.tools.base import ToolError
    from mini_agent.tools.registry import default_registry

    write(workdir, "a.py", "x = 1\ny = 2\n")
    registry = default_registry()
    tool, args = registry.parse("edit_file", {"path": "a.py", "old_string": "y = 2", "new_string": "y = 3"})
    preview = tool.preview(args, ctx)
    assert "-y = 2" in preview and "+y = 3" in preview
    assert (workdir / "a.py").read_text() == "x = 1\ny = 2\n"  # preview never writes

    tool, args = registry.parse("edit_file", {"path": "a.py", "old_string": "nope", "new_string": "y"})
    with pytest.raises(ToolError):
        tool.preview(args, ctx)
