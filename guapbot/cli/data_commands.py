"""
guapbot/cli/data_commands.py

CLI commands for data management.
Registered in cli/main.py under the 'data' group.

Commands:
    guapbot data download           — full history for all pairs/intervals
    guapbot data download XBTUSD    — full history for one pair, all intervals
    guapbot data info               — show cache summary table
    guapbot data fetch XBTUSD 1h    — incremental update for one pair/interval
"""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from guapbot.data.manager import DataManager, ALL_PAIRS, ALL_INTERVALS
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

data_app = typer.Typer(name="data", help="Data download and cache management.")


@data_app.command("download")
def download(
    pair: Optional[str] = typer.Argument(
        None,
        help="Pair to download (e.g. XBTUSD). Omit for all pairs.",
    ),
    interval: Optional[str] = typer.Option(
        None,
        "--interval", "-i",
        help="Interval: 1h, 4h, 1d. Omit for all intervals.",
    ),
) -> None:
    """Download full OHLC history from Kraken and cache to Parquet."""
    manager = DataManager()

    pairs = [pair.upper()] if pair else ALL_PAIRS
    intervals = [interval] if interval else ALL_INTERVALS

    # Validate inputs
    for p in pairs:
        if p not in ALL_PAIRS:
            console.print(f"[red]Unknown pair '{p}'. Valid: {ALL_PAIRS}[/red]")
            raise typer.Exit(1)
    for iv in intervals:
        if iv not in ALL_INTERVALS:
            console.print(f"[red]Unknown interval '{iv}'. Valid: {ALL_INTERVALS}[/red]")
            raise typer.Exit(1)

    total = len(pairs) * len(intervals)
    console.print(f"[bold]GuapBot Data Download[/bold] — {total} dataset(s)")

    count = 0
    errors = 0
    for p in pairs:
        for iv in intervals:
            count += 1
            console.print(f"  [{count}/{total}] {p} {iv}...", end=" ")
            try:
                df = manager.full_history(p, iv)
                console.print(
                    f"[green]✓[/green] {len(df):,} bars  "
                    f"({df.index[0].date()} → {df.index[-1].date()})"
                )
            except Exception as exc:
                console.print(f"[red]✗ ERROR: {exc}[/red]")
                logger.error(f"Download failed {p} {iv}: {exc}")
                errors += 1

    if errors:
        console.print(f"\n[yellow]Completed with {errors} error(s).[/yellow]")
    else:
        console.print("\n[bold green]All downloads complete.[/bold green]")

    # Show summary table
    _print_cache_table(manager)


@data_app.command("fetch")
def fetch(
    pair: str = typer.Argument(..., help="Pair: XBTUSD, ETHUSD, ETHBTC"),
    interval: str = typer.Argument("1h", help="Interval: 1h, 4h, 1d"),
) -> None:
    """Incremental update — fetch only new bars since last cache entry."""
    manager = DataManager()
    console.print(f"Fetching {pair.upper()} {interval}...")
    try:
        df = manager.fetch(pair.upper(), interval)
        console.print(f"[green]✓[/green] {len(df):,} bars total (up to {df.index[-1]})")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)


@data_app.command("info")
def info() -> None:
    """Show a summary of all cached data."""
    manager = DataManager()
    _print_cache_table(manager)


def _print_cache_table(manager: DataManager) -> None:
    cache = manager.cache_info()
    if not cache:
        console.print("[yellow]No cached data found.[/yellow]")
        return

    table = Table(title="GuapBot Cache", show_header=True, header_style="bold cyan")
    table.add_column("Dataset", style="bold")
    table.add_column("Bars", justify="right")
    table.add_column("First Bar")
    table.add_column("Last Bar")
    table.add_column("Size")

    for name, meta in sorted(cache.items()):
        if "error" in meta:
            table.add_row(name, "—", "—", "—", f"[red]{meta['error']}[/red]")
        else:
            table.add_row(
                name,
                f"{meta['bars']:,}",
                str(meta["first"])[:10],
                str(meta["last"])[:10],
                f"{meta['size_kb']} KB",
            )

    console.print(table)
