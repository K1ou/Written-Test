"""Path sandbox for the file tools.

Every path the model supplies is joined to the working directory, resolved with realpath
(which follows symlinks and, on Windows, junctions) and only accepted if the resolved
location is still inside the working directory. The comparison uses commonpath on whole
path components, so a sibling like /work-evil is not mistaken for /work.
"""

from __future__ import annotations

import os
from pathlib import Path

from mini_agent.tools.base import ToolError


class SandboxError(ToolError):
    pass


def is_within(root_real: str, target_real: str) -> bool:
    root = os.path.normcase(root_real)
    target = os.path.normcase(target_real)
    try:
        return os.path.commonpath([root, target]) == root
    except ValueError:  # different drives on Windows
        return False


class Sandbox:
    def __init__(self, workdir: str | os.PathLike[str]) -> None:
        real = os.path.realpath(workdir)
        if not os.path.isdir(real):
            raise ValueError(f"Working directory does not exist or is not a directory: {workdir}")
        self.root = Path(real)

    def resolve(self, user_path: str) -> Path:
        """Return the real absolute path for `user_path`, or raise SandboxError if it escapes."""
        if "\0" in user_path:
            raise SandboxError("Invalid path: contains a NUL byte.")
        real = os.path.realpath(os.path.join(self.root, user_path))
        if not is_within(str(self.root), real):
            raise SandboxError(
                f"Path {user_path!r} is outside the working directory. "
                "Only paths inside the working directory can be accessed."
            )
        return Path(real)

    def contains(self, real_path: str | os.PathLike[str]) -> bool:
        return is_within(str(self.root), os.path.realpath(real_path))

    def relative(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            rel = Path(os.path.relpath(path, self.root))
        return rel.as_posix() or "."
