"""
guapbot/cli/features_commands.py

CLI commands for the feature layer.

Commands:
    guapbot features build [PAIR]   — build features for one or all pairs
    guapbot features info           — show cache status for all pairs
    guapbot features inspect PAIR   — print the latest observation vector

Workflow:
    1. guapbot data download          ← ensure OHLCV is up to date (Session 2)
    2. guapbot features build         ← build all feature caches
    3. guapbot features info          ← verify everything looks right
    4. guapbot features inspect XBTUSD ← sanity check latest observation
"""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

features_app = typer.Typer(
    name="features",
    help="Feature construction and cache management.",
)

_ALL_PAIRS = ["XBTUSD", "ETHUSD", "ETHBTC"]


@features_app.command("build")
def build(
    pair: Optional[str] = typer.Argument(
        None,
        help="Pair to build features for (XBTUSD, ETHUSD, ETHBTC). Omit for all.",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Rebuild from scratch even if cache exists.",
    ),
) -> None:
    """
    Build normalised feature cache for one or all pairs.

    Runs the full pipeline: OHLCV → multi-timeframe indicators →
    cross-asset injection → rolling z-score normalisation → Parquet save.
    """
    from guapbot.features.pipeline import FeaturePipeline

    pairs = [pair.upper()] if pair else _ALL_PAIRS

    for p in pairs:
        if p not in _ALL_PAIRS:
            console.print(f"[red]Unknown pair '{p}'. Valid: {_ALL_PAIRS}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]GuapBot Feature Builder[/bold] — building {len(pairs)} pair(s)\n")
    pipe = FeaturePipeline()

    total_errors = 0
    for i, p in enumerate(pairs, 1):
        console.print(f"[{i}/{len(pairs)}] [bold]{p}[/bold]...", end=" ")

        # Check if cache exists and skip unless forced
        if not force:
            try:
                existing = pipe.transform(p)
                console.print(
                    f"[yellow]skipped[/yellow] (cache exists: "
                    f"{len(existing):,} bars, {len(existing.columns)} features). "
                    f"Use --force to rebuild."
                )
                continue
            except FileNotFoundError:
                pass  # No cache — proceed to build

        try:
            df = pipe.fit_transform(p)
            console.print(
                f"[green]✓[/green] "
                f"{len(df):,} bars · "
                f"{len(df.columns)} features · "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
        except Exception as exc:
            console.print(f"[red]✗ ERROR: {exc}[/red]")
            logger.exception(f"Feature build failed for {p}")
            total_errors += 1

    console.print()
    if total_errors:
        console.print(f"[yellow]Completed with {total_errors} error(s).[/yellow]")
        console.print("Check logs for details. Common causes:")
        console.print("  - No OHLCV data: run [bold]guapbot data download[/bold] first")
        console.print("  - Insufficient bars: need at least 200 bars of 1h data")
    else:
        console.print("[bold green]All features built successfully.[/bold green]")

    console.print()
    _print_feature_table(pipe)


@features_app.command("info")
def info() -> None:
    """Show feature cache status for all pairs."""
    from guapbot.features.pipeline import FeaturePipeline
    pipe = FeaturePipeline()
    _print_feature_table(pipe)


@features_app.command("inspect")
def inspect(
    pair: str = typer.Argument(..., help="Pair to inspect (XBTUSD, ETHUSD, ETHBTC)"),
    n: int = typer.Option(10, "--n", help="Number of features to show in preview"),
    full: bool = typer.Option(False, "--full", help="Show all features"),
) -> None:
    """
    Print the latest observation vector for a pair.

    Shows the current feature values that would be fed into the models.
    Use this to sanity-check the feature pipeline after building.
    """
    from guapbot.features.pipeline import FeaturePipeline

    pair = pair.upper()
    if pair not in _ALL_PAIRS:
        console.print(f"[red]Unknown pair '{pair}'. Valid: {_ALL_PAIRS}[/red]")
        raise typer.Exit(1)

    pipe = FeaturePipeline()

    try:
        obs = pipe.get_observation(pair)
    except FileNotFoundError:
        console.print(
            f"[red]No feature cache for {pair}. "
            f"Run: guapbot features build {pair}[/red]"
        )
        raise typer.Exit(1)

    ts = obs.pop("timestamp")
    features = list(obs.items())

    console.print(f"\n[bold]Latest observation: {pair}[/bold]")
    console.print(f"Timestamp: [cyan]{ts}[/cyan]")
    console.print(f"Total features: [cyan]{len(features)}[/cyan]\n")

    table = Table(
        title=f"{pair} Feature Vector (latest bar)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Feature", style="bold", width=40)
    table.add_column("Value", justify="right", width=15)
    table.add_column("Signal", justify="center", width=10)

    display = features if full else features[:n]

    for name, val in display:
        try:
            fval = float(val)
            val_str = f"{fval:.4f}"
            if fval > 0.5:
                signal = "[green]▲[/green]"
            elif fval < -0.5:
                signal = "[red]▼[/red]"
            else:
                signal = "[yellow]–[/yellow]"
        except (TypeError, ValueError):
            val_str = str(val)
            signal = ""

        table.add_row(name, val_str, signal)

    console.print(table)

    if not full and len(features) > n:
        console.print(
            f"\n[dim]Showing {n} of {len(features)} features. "
            f"Use --full to see all.[/dim]"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_feature_table(pipe) -> None:
    """Print a Rich table summarising the feature cache."""
    info = pipe.cache_info()

    table = Table(
        title="GuapBot Feature Cache",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Pair", style="bold")
    table.add_column("Bars", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("First Bar")
    table.add_column("Last Bar")
    table.add_column("Norm Size")
    table.add_column("Raw Size")

    for pair in _ALL_PAIRS:
        meta = info.get(pair, {})
        if "error" in meta:
            table.add_row(pair, "—", "—", "—", "—", f"[red]{meta['error']}[/red]", "—")
        elif "status" in meta:
            table.add_row(
                pair, "—", "—", "—", "—",
                f"[yellow]{meta['status']}[/yellow]", "—"
            )
        else:
            table.add_row(
                pair,
                f"{meta.get('bars', 0):,}",
                f"{meta.get('features', 0):,}",
                str(meta.get("first", "—"))[:10],
                str(meta.get("last", "—"))[:10],
                f"{meta.get('norm_size_kb', '—')} KB",
                f"{meta.get('raw_size_kb', '—')} KB",
            )

    console.print(table)
