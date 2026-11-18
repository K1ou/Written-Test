import pytest

from mini_agent.tools.base import Tool, ToolArgs, ToolContext, ToolError
from mini_agent.tools.registry import ToolRegistry, default_registry
from mini_agent.tools.truncate import HeadTailBuffer, truncate_chars


def put(workdir, name, text="x\n"):
    path = workdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# ---------------------------------------------------------------- list_files


def test_list_directory_marks_dirs_and_ignores_noise_dirs(run_tool, workdir):
    put(workdir, "a.py")
    put(workdir, "src/b.py")
    for ignored in (".git", "node_modules", "__pycache__", ".venv"):
        put(workdir, f"{ignored}/junk.txt")
    result = run_tool("list_files")
    assert result.content.splitlines() == ["a.py", "src/"]


def test_list_with_glob_is_recursive_and_skips_ignored_dirs(run_tool, workdir):
    put(workdir, "a.py")
    put(workdir, "src/b.py")
    put(workdir, "src/deep/c.py")
    put(workdir, "src/readme.md")
    put(workdir, "node_modules/pkg/d.py")
    assert run_tool("list_files", pattern="*.py").content.splitlines() == ["a.py", "src/b.py", "src/deep/c.py"]
    assert run_tool("list_files", pattern="src/*.py").content.splitlines() == ["src/b.py"]
    assert run_tool("list_files", pattern="src/**/*.py").content.splitlines() == ["src/b.py", "src/deep/c.py"]
    assert run_tool("list_files", pattern="*.rs").content == "No files matched the pattern."


def test_list_results_are_capped_with_a_note(run_tool, workdir):
    for i in range(250):
        put(workdir, f"f{i:03}.txt")
    result = run_tool("list_files")
    assert len(result.content.splitlines()) == 201
    assert "250 entries in total" in result.content


def test_list_errors(run_tool, workdir):
    put(workdir, "a.txt")
    assert run_tool("list_files", path="a.txt").is_error
    assert run_tool("list_files", path="missing").is_error


# ----------------------------------------------------------------- read_file


def test_read_returns_numbered_lines_and_supports_ranges(run_tool, workdir):
    put(workdir, "a.txt", "one\ntwo\nthree\nfour\n")
    assert run_tool("read_file", path="a.txt").content == "     1\tone\n     2\ttwo\n     3\tthree\n     4\tfour"
    assert run_tool("read_file", path="a.txt", start_line=2, end_line=3).content == "     2\ttwo\n     3\tthree"
    assert run_tool("read_file", path="a.txt", start_line=4).content == "     4\tfour"


def test_read_strips_carriage_returns(run_tool, workdir):
    put(workdir, "a.txt", "one\r\ntwo\r\n")
    assert "\r" not in run_tool("read_file", path="a.txt").content


def test_read_errors(run_tool, workdir):
    put(workdir, "a.txt", "one\ntwo\n")
    put(workdir, "empty.txt", "")
    (workdir / "bin.dat").write_bytes(b"abc\x00def")
    (workdir / "d").mkdir()
    assert "only 2 lines" in run_tool("read_file", path="a.txt", start_line=5).content
    assert "smaller than start_line" in run_tool("read_file", path="a.txt", start_line=2, end_line=1).content
    assert run_tool("read_file", path="empty.txt").content == "(empty file)"
    assert "binary" in run_tool("read_file", path="bin.dat").content
    assert "directory" in run_tool("read_file", path="d").content
    assert "not found" in run_tool("read_file", path="nope.txt").content.lower()


def test_large_file_is_truncated_and_can_be_continued_by_line_range(run_tool, workdir):
    put(workdir, "big.log", "".join(f"line {i}\n" for i in range(1, 100_001)))
    first = run_tool("read_file", path="big.log")
    assert not first.is_error
    assert "100000 lines" in first.content and "start_line=501" in first.content
    assert first.content.count("\n") < 510
    second = run_tool("read_file", path="big.log", start_line=501)
    assert second.content.splitlines()[0] == "   501\tline 501"
    assert "start_line=1001" in second.content


def test_five_megabyte_single_line_file_does_not_flood_the_context(run_tool, workdir):
    put(workdir, "huge.log", "x" * 5_000_000)
    result = run_tool("read_file", path="huge.log")
    assert len(result.content) < 3_000
    assert "line truncated" in result.content and "5000000" in result.content


def test_char_budget_cuts_before_the_line_limit(run_tool, workdir):
    put(workdir, "wide.txt", "".join("y" * 100 + "\n" for _ in range(500)))
    result = run_tool("read_file", path="wide.txt")
    assert len(result.content) < 31_000
    assert "Truncated" in result.content and "start_line=" in result.content


# --------------------------------------------------------------- write_file


def test_write_creates_parent_directories(run_tool, workdir):
    result = run_tool("write_file", path="a/b/c.txt", content="hi\n")
    assert not result.is_error and "Created" in result.content
    assert (workdir / "a" / "b" / "c.txt").read_text() == "hi\n"


def test_write_overwrites_and_reports_it(run_tool, workdir):
    put(workdir, "a.txt", "old")
    result = run_tool("write_file", path="a.txt", content="new")
    assert "Overwrote" in result.content and (workdir / "a.txt").read_text() == "new"


def test_write_keeps_newlines_exactly_and_utf8(run_tool, workdir):
    run_tool("write_file", path="a.txt", content="你好\nline2\n")
    assert (workdir / "a.txt").read_bytes() == "你好\nline2\n".encode()


def test_write_to_directory_is_an_error(run_tool, workdir):
    (workdir / "d").mkdir()
    assert run_tool("write_file", path="d", content="x").is_error


def test_write_preview_shows_diff_or_new_file(workdir, ctx):
    registry = default_registry()
    put(workdir, "a.txt", "old\n")
    tool, args = registry.parse("write_file", {"path": "a.txt", "content": "new\n"})
    assert "-old" in tool.preview(args, ctx) and "+new" in tool.preview(args, ctx)
    tool, args = registry.parse("write_file", {"path": "fresh.txt", "content": "a\nb\n"})
    preview = tool.preview(args, ctx)
    assert preview.startswith("Create fresh.txt") and "+a" in preview
    assert not (workdir / "fresh.txt").exists()


# ---------------------------------------------------------------------- grep


def test_grep_returns_path_line_and_text(run_tool, workdir):
    put(workdir, "a.py", "import os\nTODO: fix\n")
    put(workdir, "src/b.py", "x = 1\n# TODO later\n")
    result = run_tool("grep", pattern="TODO")
    assert result.content.splitlines() == ["a.py:2:TODO: fix", "src/b.py:2:# TODO later"]


def test_grep_options_and_targets(run_tool, workdir):
    put(workdir, "a.py", "Hello\n")
    put(workdir, "b.txt", "hello\n")
    assert run_tool("grep", pattern="hello").content == "b.txt:1:hello"
    assert len(run_tool("grep", pattern="hello", ignore_case=True).content.splitlines()) == 2
    assert run_tool("grep", pattern="hello", ignore_case=True, include="*.py").content == "a.py:1:Hello"
    assert run_tool("grep", pattern="Hello", path="a.py").content == "a.py:1:Hello"
    assert run_tool("grep", pattern="zzz").content == "No matches found."


def test_grep_invalid_regex_is_an_error_not_a_crash(run_tool, workdir):
    result = run_tool("grep", pattern="(unclosed")
    assert result.is_error and "Invalid regular expression" in result.content


def test_grep_skips_binary_and_ignored_directories(run_tool, workdir):
    (workdir / "bin.dat").write_bytes(b"needle\x00")
    put(workdir, "node_modules/p/x.js", "needle\n")
    put(workdir, "ok.txt", "needle\n")
    assert run_tool("grep", pattern="needle").content == "ok.txt:1:needle"


def test_grep_caps_matches_and_long_lines(run_tool, workdir):
    put(workdir, "many.txt", "hit\n" * 300)
    result = run_tool("grep", pattern="hit")
    assert len(result.content.splitlines()) == 101 and "300 matches in total" in result.content
    put(workdir, "long.txt", "hit" + "z" * 1000 + "\n")
    assert len(run_tool("grep", pattern="hit", path="long.txt").content) < 400


# ------------------------------------------------------- registry & validation


def test_unknown_tool_lists_available_tools_and_suggests_a_close_name():
    with pytest.raises(ToolError) as info:
        default_registry().parse("read_fil", {})
    assert "Unknown tool" in str(info.value) and "read_file" in str(info.value) and "Did you mean" in str(info.value)


def test_missing_required_field_names_the_field():
    with pytest.raises(ToolError) as info:
        default_registry().parse("read_file", {})
    assert "missing required field 'path'" in str(info.value)


def test_wrong_type_and_unexpected_field_are_reported():
    with pytest.raises(ToolError) as info:
        default_registry().parse("read_file", {"path": "a", "start_line": "abc", "bogus": 1})
    message = str(info.value)
    assert "start_line" in message and "'abc'" in message and "unexpected field 'bogus'" in message


def test_numeric_strings_are_coerced(run_tool, workdir):
    put(workdir, "a.txt", "one\ntwo\n")
    assert run_tool("read_file", path="a.txt", start_line="2").content == "     2\ttwo"


def test_start_line_below_one_is_rejected(run_tool, workdir):
    put(workdir, "a.txt", "one\n")
    assert run_tool("read_file", path="a.txt", start_line=0).is_error


def schema_keys(node) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            if key == "properties":
                for prop in value.values():
                    keys |= schema_keys(prop)
            else:
                keys |= schema_keys(value)
    elif isinstance(node, list):
        for item in node:
            keys |= schema_keys(item)
    return keys


def test_tool_schemas_are_clean_json_schema_for_every_vendor():
    specs = {s.name: s for s in default_registry().specs()}
    assert set(specs) == {"list_files", "read_file", "write_file", "edit_file", "grep", "run_shell"}
    for spec in specs.values():
        assert spec.description
        assert spec.parameters["type"] == "object"
        assert not schema_keys(spec.parameters) & {"title", "default", "additionalProperties", "anyOf", "$defs"}
    assert specs["edit_file"].parameters["required"] == ["path", "old_string", "new_string"]
    assert specs["read_file"].parameters["properties"]["end_line"]["type"] == "integer"
    assert "required" not in specs["list_files"].parameters or "path" not in specs["list_files"].parameters["required"]


def test_confirmation_flags():
    registry = default_registry()
    flags = {name: registry.get(name).requires_confirmation for name in registry.names}
    assert flags == {
        "list_files": False,
        "read_file": False,
        "write_file": True,
        "edit_file": True,
        "grep": False,
        "run_shell": True,
    }


def test_a_crashing_tool_becomes_an_error_result(ctx):
    class Boom(Tool):
        name = "boom"
        description = "x"
        params = ToolArgs

        def run(self, args, ctx):
            raise RuntimeError("kaput")

    registry = ToolRegistry([Boom()])
    tool, args = registry.parse("boom", {})
    result = registry.execute(tool, args, ctx)
    assert result.is_error and "RuntimeError" in result.content and "kaput" in result.content


def test_every_result_is_capped_as_a_last_resort(ctx):
    class Flood(Tool):
        name = "flood"
        description = "x"
        params = ToolArgs

        def run(self, args, ctx):
            return "line\n" * 100_000

    registry = ToolRegistry([Flood()])
    tool, args = registry.parse("flood", {})
    result = registry.execute(tool, args, ctx)
    assert len(result.content) < 31_000 and "Truncated" in result.content


# ------------------------------------------------------------------ truncation


def test_truncate_chars_reports_totals():
    text = "\n".join(f"l{i}" for i in range(1000))
    out = truncate_chars(text, limit=100)
    assert "1000 lines" in out and out.startswith("l0\n")
    assert truncate_chars("short") == "short"


def test_head_tail_buffer_keeps_both_ends_and_counts_everything():
    buf = HeadTailBuffer(head_bytes=20, tail_bytes=30)
    for i in range(1000):
        buf.write(f"line {i}\n".encode())
    text = buf.render()
    assert text.startswith("line 0\nline 1\n")
    assert text.rstrip().endswith("line 999")
    assert "bytes omitted" in text and "1000 lines" in text
    assert buf.total_lines == 1000


def test_head_tail_buffer_returns_small_output_untouched():
    buf = HeadTailBuffer()
    buf.write(b"hello ")
    buf.write("世界".encode())
    assert buf.render() == "hello 世界" and not buf.truncated
