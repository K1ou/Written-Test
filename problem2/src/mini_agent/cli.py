from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="Mini Code Agent")


@app.callback(invoke_without_command=True)
def main(
    workdir: Annotated[str, typer.Option(help="Working directory the agent may access")] = ".",
) -> None:
    typer.echo(f"mini-agent: not implemented yet (workdir={workdir})")
