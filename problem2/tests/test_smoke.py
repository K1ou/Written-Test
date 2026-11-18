from typer.testing import CliRunner

from mini_agent.cli import app


def test_cli_help_runs():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--workdir" in result.output
