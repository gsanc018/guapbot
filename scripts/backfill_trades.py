"""
scripts/backfill_trades.py

Backfill OHLCV bars from Kraken raw tick data for one or all pairs.

This script closes the gap between the last Kraken quarterly CSV dump (2025-12-31)
and today by fetching raw tick trades from GET /public/Trades and aggregating them
into 1h/4h/1d OHLCV bars.

Usage:
    # Backfill one pair (resumes from last cached bar by default)
    python scripts/backfill_trades.py --pair XBTUSD

    # Backfill all pairs
    python scripts/backfill_trades.py --all-pairs

    # Start from a specific date (Unix timestamp in seconds)
    python scripts/backfill_trades.py --pair XBTUSD --since-ts 1735689600

    # Dry run: fetch first page only, print stats, don't save
    python scripts/backfill_trades.py --pair XBTUSD --dry-run

Warning:
    Fetching 2 months of BTC ticks ≈ 10M trades ≈ 10,000 API pages.
    Expect roughly 1 hour per pair at Kraken's ~3 req/s public rate limit.
    For subsequent runs (weekly/monthly), only the gap since last bar is fetched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

# Make guapbot importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from guapbot.data.aggregator import rows_to_tick_df, aggregate_trades_to_ohlcv
from guapbot.data.kraken_client import KrakenClient
from guapbot.data.manager import DataManager, ALL_PAIRS
from guapbot.data.store import ParquetStore
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer(help="Backfill OHLCV bars from Kraken raw tick data.")


def _ts_to_ns(unix_seconds: float) -> int:
    """Convert Unix seconds to nanoseconds (Kraken Trades cursor format)."""
    return int(unix_seconds * 1_000_000_000)


@app.command()
def main(
    pair: Optional[str] = typer.Option(
        None, "--pair", help="Single pair to backfill: XBTUSD, ETHUSD, or ETHBTC"
    ),
    all_pairs: bool = typer.Option(
        False, "--all-pairs", help="Run backfill for all configured pairs"
    ),
    since_ts: Optional[int] = typer.Option(
        None,
        "--since-ts",
        help=(
            "Unix timestamp (seconds) to start backfill from. "
            "Defaults to the last cached bar's timestamp. "
            "Pass 0 to fetch the full history from the very beginning (very slow)."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Fetch the first page only, print stats, and exit without saving.",
    ),
) -> None:
    """
    Backfill OHLCV bars by fetching raw tick trades from Kraken's REST API.

    Fetches all trades since the last cached bar (or --since-ts), aggregates them
    into 1h / 4h / 1d OHLCV bars, and appends the result to the local Parquet cache.
    """
    if pair is None and not all_pairs:
        typer.echo("Error: specify --pair XBTUSD or --all-pairs", err=True)
        raise typer.Exit(1)

    if pair is not None and pair not in ALL_PAIRS:
        typer.echo(f"Error: unknown pair '{pair}'. Valid: {ALL_PAIRS}", err=True)
        raise typer.Exit(1)

    pairs_to_run = ALL_PAIRS if all_pairs else [pair]  # type: ignore[list-item]

    store = ParquetStore()
    client = KrakenClient()
    manager = DataManager(client=client, store=store)

    for p in pairs_to_run:
        typer.echo(f"\n{'='*50}")
        typer.echo(f"Pair: {p}")

        # Determine the nanosecond cursor to start from
        if since_ts is not None:
            since_cursor = _ts_to_ns(since_ts)
            typer.echo(f"Starting from --since-ts: {since_ts} ({since_cursor} ns)")
        else:
            last_ts = store.last_timestamp(p, "1h")
            if last_ts is not None:
                since_cursor = _ts_to_ns(last_ts.timestamp())
                typer.echo(f"Resuming from last cached bar: {last_ts} ({since_cursor} ns)")
            else:
                since_cursor = 0
                typer.echo(
                    "No cache found — fetching full history from Kraken (this will be slow)"
                )

        if dry_run:
            typer.echo(f"[DRY RUN] Would backfill {p} from cursor={since_cursor}")
            rows, last = client.get_trades(pair=p, since=since_cursor)
            typer.echo(f"[DRY RUN] First page: {len(rows)} trades returned, last cursor={last}")
            if rows:
                ticks = rows_to_tick_df(rows)
                typer.echo(
                    f"[DRY RUN] Tick time range: "
                    f"{ticks['timestamp'].min()} → {ticks['timestamp'].max()}"
                )
                for interval in ["1h", "4h", "1d"]:
                    sample = aggregate_trades_to_ohlcv(ticks, interval)
                    typer.echo(f"[DRY RUN]   {interval}: {len(sample)} bars from this page")
            continue

        # Full backfill
        results = manager.backfill_trades(pair=p, since=since_cursor)

        if results:
            typer.echo(f"Backfill complete for {p}:")
            for interval, df in results.items():
                first = df.index[0].date() if not df.empty else "N/A"
                last = df.index[-1].date() if not df.empty else "N/A"
                typer.echo(f"  {interval}: {len(df)} total bars cached  ({first} → {last})")
        else:
            typer.echo(f"No new bars added for {p} — cache is already up to date.")

    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
