"""
guapbot/cli/main.py

Main CLI entry point. All guapbot commands registered here.

Session 1 foundation commands are defined inline.
Each session adds its own command module and registers it here.
"""
import typer
from rich.console import Console

from guapbot.cli.data_commands import data_app

app = typer.Typer(
    name="guapbot",
    help="GuapBot — Multi-Strategy Crypto Trading System",
    no_args_is_help=True,
)
console = Console()

# Register sub-applications
app.add_typer(data_app, name="data")


@app.command()
def version() -> None:
    """Show GuapBot version."""
    from guapbot import __version__
    console.print(f"GuapBot [bold]{__version__}[/bold]")


@app.command()
def status() -> None:
    """Show system status (placeholder — full implementation in Session 7)."""
    console.print("[yellow]Status command available from Session 7 (execution layer)[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
