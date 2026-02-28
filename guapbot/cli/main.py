"""
guapbot/cli/main.py

Main CLI entry point. All guapbot commands registered here.

Session 1 foundation commands are defined inline.
Each session adds its own command module and registers it here.
"""
from pathlib import Path

import typer
from rich.console import Console

from guapbot.cli.backtest_commands import backtest_app
from guapbot.cli.data_commands import data_app
from guapbot.cli.features_commands import features_app
from guapbot.cli.model_commands import model_app
from guapbot.cli.paper_commands import paper_app

app = typer.Typer(
    name="guapbot",
    help="GuapBot — Multi-Strategy Crypto Trading System",
    no_args_is_help=True,
)
console = Console()

# Register sub-applications
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(model_app, name="models")
app.add_typer(backtest_app, name="backtest")
app.add_typer(paper_app, name="paper")


@app.command("train")
def train(
    model_type: str = typer.Argument(..., help="Model type: trend | meanrev | xgb | lstm | rl"),
    pair: str = typer.Argument(..., help="Pair: XBTUSD or ETHUSD"),
    strategy: str = typer.Option("money_printer", "--strategy", "-s"),
    algo: str = typer.Option("sac", "--algo"),
    train_end: str = typer.Option("2024-06-30", "--train-end"),
    val_end: str = typer.Option("2024-12-31", "--val-end"),
) -> None:
    """Train a model (shortcut for guapbot models train)."""
    from guapbot.cli.model_commands import train as _train
    _train(
        model_type=model_type,
        pair=pair,
        strategy=strategy,
        algo=algo,
        train_end=train_end,
        val_end=val_end,
        save_dir=Path("models"),
    )


@app.command()
def version() -> None:
    """Show GuapBot version."""
    from guapbot import __version__
    console.print(f"GuapBot [bold]{__version__}[/bold]")


@app.command()
def status() -> None:
    """Show system status: Redis connection, trading mode, active models."""
    from guapbot.utils.config import settings
    from guapbot.execution.market_state import MarketState

    mode_colour = "green" if settings.is_paper else "red"
    console.print(f"  Trading mode : [{mode_colour}]{settings.trading_mode.upper()}[/{mode_colour}]")

    state = MarketState()
    state.connect()
    redis_colour = "green" if state.connected else "red"
    console.print(f"  Redis        : [{redis_colour}]{'connected' if state.connected else 'offline'}[/{redis_colour}]")

    if state.connected:
        keys = state.all_keys()
        console.print(f"  State keys   : {len(keys)}")
        state.disconnect()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
