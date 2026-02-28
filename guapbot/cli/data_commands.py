"""
guapbot/cli/data_commands.py

Workflow:
    1. guapbot data import --dir data/raw/master_q4   ← Kraken quarterly CSVs into Parquet
    2. guapbot data download                           ← backfill recent bars via tick API
    3. guapbot data info                               ← check what's cached
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from guapbot.data.manager import DataManager, ALL_PAIRS, ALL_INTERVALS
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()
data_app = typer.Typer(name="data", help="Data download and cache management.")


@data_app.command("import")
def import_csv(
    dir: Path = typer.Option(
        Path("data/raw/master_q4"),
        "--dir", "-d",
        help="Directory containing Kraken OHLCVT CSV files.",
    ),
) -> None:
    """Import Kraken OHLCVT CSV files into the Parquet cache."""
    from guapbot.data.importer import KrakenCSVImporter

    if not dir.exists():
        console.print(f"[red]Directory not found: {dir}[/red]")
        console.print("Usage: guapbot data import --dir data/raw/master_q4")
        raise typer.Exit(1)

    console.print(f"[bold]GuapBot Data Import[/bold] — reading CSVs from {dir}")
    importer = KrakenCSVImporter()

    try:
        results = importer.import_dir(dir)
    except Exception as exc:
        console.print(f"[red]Import failed: {exc}[/red]")
        raise typer.Exit(1)

    if not results:
        console.print("[yellow]No matching CSV files found.[/yellow]")
        console.print("Expected files like: XBTUSD_60.csv, ETHUSD_240.csv, XETHXXBT_1440.csv")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Imported {len(results)} dataset(s).[/bold green]")
    console.print("Run [bold]guapbot data download[/bold] to pull the latest bars from Kraken.\n")
    _print_cache_table(DataManager())


@data_app.command("download")
def download(
    pair: Optional[str] = typer.Argument(None, help="Pair to backfill. Omit for all."),
    since: Optional[int] = typer.Option(
        None,
        "--since",
        help=(
            "Start nanosecond cursor. Defaults to resuming from the last cached bar. "
            "Pass 0 to re-fetch from the very beginning (slow)."
        ),
    ),
) -> None:
    """
    Backfill bars from Kraken tick data, resuming from the last cached bar.

    Fetches raw trades via GET /public/Trades, aggregates to 1h/4h/1d bars,
    and appends new data to the local Parquet cache for all intervals at once.

    Warning: first-time backfill of 2+ months of BTC ticks takes ~1 hour per pair.
    Subsequent runs (incremental) are fast — only fetches ticks since the last bar.
    """
    manager = DataManager()
    pairs = [pair.upper()] if pair else ALL_PAIRS

    for p in pairs:
        if p not in ALL_PAIRS:
            console.print(f"[red]Unknown pair '{p}'. Valid: {ALL_PAIRS}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]GuapBot Data Download[/bold] — backfilling {len(pairs)} pair(s) from Kraken")

    errors = 0
    for i, p in enumerate(pairs, 1):
        console.print(f"\n  [{i}/{len(pairs)}] {p}...")
        try:
            kwargs = {"since": since} if since is not None else {}
            results = manager.backfill_trades(p, **kwargs)
            if results:
                for iv, df in results.items():
                    last = df.index[-1].date() if not df.empty else "N/A"
                    console.print(f"    [green]✓[/green] {iv}: {len(df):,} bars cached  (latest: {last})")
            else:
                console.print(f"    [yellow]No new bars — cache already up to date.[/yellow]")
        except Exception as exc:
            console.print(f"    [red]✗ ERROR: {exc}[/red]")
            errors += 1

    if errors:
        console.print(f"\n[yellow]Completed with {errors} error(s).[/yellow]")
    else:
        console.print("\n[bold green]Download complete.[/bold green]")
    _print_cache_table(manager)


@data_app.command("fetch")
def fetch(
    pair: str = typer.Argument(..., help="Pair: XBTUSD, ETHUSD, ETHBTC"),
    interval: str = typer.Argument("1h", help="Interval: 1h, 4h, 1d"),
) -> None:
    """Show cached bars for a single pair/interval (reads cache only, no API call)."""
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
    _print_cache_table(DataManager())


def _print_cache_table(manager: DataManager) -> None:
    cache = manager.cache_info()
    if not cache:
        console.print("[yellow]No cached data found. Run: guapbot data import --dir data/raw/master_q4[/yellow]")
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
