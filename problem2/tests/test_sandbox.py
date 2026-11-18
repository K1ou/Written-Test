import os
import subprocess
import sys

import pytest

from mini_agent.tools.sandbox import Sandbox, SandboxError

IS_WINDOWS = sys.platform == "win32"


def link_dir(link, target) -> bool:
    """Create a directory symlink, or a junction on Windows when symlinks are not permitted."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if IS_WINDOWS:
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        return result.returncode == 0
    return False


@pytest.fixture
def layout(tmp_path):
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    (work / "a.txt").write_text("a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    evil = tmp_path / "work-evil"
    evil.mkdir()
    (evil / "x.txt").write_text("x")
    return work, outside, evil


@pytest.fixture
def sandbox(layout):
    return Sandbox(layout[0])


def test_relative_paths_inside_are_allowed(sandbox, layout):
    work = layout[0]
    assert sandbox.resolve("a.txt") == (work / "a.txt").resolve()
    assert sandbox.resolve(".") == work.resolve()
    assert sandbox.resolve("") == work.resolve()
    assert sandbox.resolve("sub/../a.txt") == (work / "a.txt").resolve()
    assert sandbox.resolve("sub/new/file.txt") == (work / "sub" / "new" / "file.txt").resolve()


def test_absolute_path_inside_is_allowed(sandbox, layout):
    assert sandbox.resolve(str(layout[0] / "a.txt")) == (layout[0] / "a.txt").resolve()


@pytest.mark.parametrize("path", ["..", "../outside/secret.txt", "sub/../../outside/secret.txt", "../../../../etc/passwd"])
def test_parent_traversal_is_rejected(sandbox, path):
    with pytest.raises(SandboxError):
        sandbox.resolve(path)


def test_absolute_path_outside_is_rejected(sandbox, layout):
    with pytest.raises(SandboxError):
        sandbox.resolve(str(layout[1] / "secret.txt"))


def test_sibling_directory_with_same_prefix_is_rejected(sandbox, layout):
    # /work-evil starts with the string /work but is not inside it.
    with pytest.raises(SandboxError):
        sandbox.resolve(str(layout[2] / "x.txt"))
    with pytest.raises(SandboxError):
        sandbox.resolve("../work-evil/x.txt")


def test_nul_byte_is_rejected(sandbox):
    with pytest.raises(SandboxError):
        sandbox.resolve("a\0.txt")


def test_symlinked_directory_pointing_outside_is_rejected(sandbox, layout):
    work, outside, _ = layout
    if not link_dir(work / "link", outside):
        pytest.skip("cannot create symlinks or junctions here")
    with pytest.raises(SandboxError):
        sandbox.resolve("link/secret.txt")
    with pytest.raises(SandboxError):
        sandbox.resolve("link/new_file.txt")  # new file below an escaping link
    with pytest.raises(SandboxError):
        sandbox.resolve("link")


def test_symlinked_file_pointing_outside_is_rejected(sandbox, layout):
    work, outside, _ = layout
    try:
        os.symlink(outside / "secret.txt", work / "file_link")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create file symlinks here")
    with pytest.raises(SandboxError):
        sandbox.resolve("file_link")


def test_symlink_staying_inside_is_allowed(sandbox, layout):
    work = layout[0]
    if not link_dir(work / "alias", work / "sub"):
        pytest.skip("cannot create symlinks or junctions here")
    assert sandbox.resolve("alias") == (work / "sub").resolve()


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows junctions")
def test_windows_junction_pointing_outside_is_rejected(sandbox, layout):
    work, outside, _ = layout
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(work / "junc"), str(outside)], capture_output=True)
    assert result.returncode == 0
    with pytest.raises(SandboxError):
        sandbox.resolve("junc/secret.txt")


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows paths are case-insensitive")
def test_windows_case_insensitive_inside_path_is_allowed(sandbox, layout):
    upper = str(layout[0]).upper()
    assert sandbox.resolve(upper + "\\A.TXT").name.lower() == "a.txt"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows drives")
def test_windows_other_drive_or_system_dir_is_rejected(sandbox):
    with pytest.raises(SandboxError):
        sandbox.resolve(os.environ.get("SystemRoot", "C:\\Windows"))


def test_missing_workdir_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        Sandbox(tmp_path / "does-not-exist")


# ---- the file tools enforce the sandbox end to end ----


def test_tools_refuse_paths_outside_the_workdir(run_tool, layout):
    outside_file = layout[1] / "secret.txt"
    for name, args in [
        ("read_file", {"path": "../outside/secret.txt"}),
        ("read_file", {"path": str(outside_file)}),
        ("list_files", {"path": ".."}),
        ("grep", {"pattern": "secret", "path": "../outside"}),
        ("write_file", {"path": "../outside/new.txt", "content": "x"}),
        ("edit_file", {"path": "../outside/secret.txt", "old_string": "secret", "new_string": "x"}),
    ]:
        result = run_tool(name, **args)
        assert result.is_error and "outside the working directory" in result.content, (name, result.content)
    assert not (layout[1] / "new.txt").exists()
    assert outside_file.read_text() == "secret"


def test_grep_does_not_follow_links_out_of_the_workdir(run_tool, layout):
    work, outside, _ = layout
    if not link_dir(work / "link", outside):
        pytest.skip("cannot create symlinks or junctions here")
    result = run_tool("grep", pattern="secret")
    assert "secret.txt" not in result.content
