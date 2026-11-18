import pytest

from mini_agent.agent.cancel import CancelToken
from mini_agent.tools.base import ToolContext, ToolError, ToolResult
from mini_agent.tools.registry import default_registry
from mini_agent.tools.sandbox import Sandbox


@pytest.fixture
def workdir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


@pytest.fixture
def ctx(workdir):
    return ToolContext(sandbox=Sandbox(workdir), cancel=CancelToken())


@pytest.fixture
def run_tool(ctx):
    """Run a tool the way the agent loop does: parse, then execute; parse errors become error results."""
    registry = default_registry()

    def _run(name: str, **args) -> ToolResult:
        try:
            tool, parsed = registry.parse(name, args)
        except ToolError as exc:
            return ToolResult(str(exc), is_error=True)
        return registry.execute(tool, parsed, ctx)

    return _run
