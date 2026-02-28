#!/usr/bin/env python3
"""
scripts/label_regimes.py
------------------------
Generate rule-based regime labels for HMM training.

Applies deterministic technical-indicator thresholds to produce
a 'label' column.  The output CSV is consumed by HMMDetector.fit().

This is fast, free, and reproducible — no API calls needed.

Usage:
    python scripts/label_regimes.py --pair XBTUSD --timeframe 1h
    python scripts/label_regimes.py --pair XBTUSD --timeframe 4h
    python scripts/label_regimes.py --pair XBTUSD --timeframe daily
    python scripts/label_regimes.py --pair XBTUSD --timeframe 1h --dry-run

After labelling, train the HMM:
    guapbot regime fit --pair XBTUSD
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import typer
from rich.console import Console

from guapbot.regime.rule_labeler import RuleLabeler
from guapbot.utils.logging import get_logger

app = typer.Typer(name="label-regimes", add_completion=False)
console = Console()
log = get_logger(__name__)

VALID_PAIRS = ["XBTUSD", "ETHUSD", "ETHBTC"]
VALID_TIMEFRAMES = ["1h", "4h", "daily"]
OUTPUT_DIR = Path("data/regime_labels")


@app.command()
def main(
    pair: str = typer.Option("XBTUSD", help="Trading pair (XBTUSD, ETHUSD, ETHBTC)"),
    timeframe: str = typer.Option("1h", help="Timeframe (1h, 4h, daily)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show label distribution without saving"
    ),
    features_dir: Path = typer.Option(
        Path("data/cache"),
        help="Directory containing features_{pair}_raw.parquet files",
    ),
    output_dir: Path = typer.Option(OUTPUT_DIR, help="Directory to save label CSV"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing labels"),
) -> None:
    """Generate rule-based regime labels for HMM training (instant, free)."""

    if pair not in VALID_PAIRS:
        console.print(f"[red]Unknown pair '{pair}'. Valid: {VALID_PAIRS}[/red]")
        raise typer.Exit(1)
    if timeframe not in VALID_TIMEFRAMES:
        console.print(f"[red]Unknown timeframe '{timeframe}'. Valid: {VALID_TIMEFRAMES}[/red]")
        raise typer.Exit(1)

    console.rule("[bold]GuapBot — Regime Labelling[/bold]")
    console.print(f"Pair: [cyan]{pair}[/cyan]  Timeframe: [cyan]{timeframe}[/cyan]")

    # ── Load raw features ──────────────────────────────────────────────
    features_path = features_dir / f"features_{pair}_raw.parquet"
    if not features_path.exists():
        console.print(
            f"[red]Features not found at {features_path}.[/red]\n"
            f"Run [bold]guapbot features build --pair {pair}[/bold] first."
        )
        raise typer.Exit(1)

    console.print(f"Loading features from {features_path} …")
    df = pd.read_parquet(features_path)
    console.print(f"Loaded [green]{len(df):,}[/green] bars ({df.index[0]} → {df.index[-1]})")

    # ── Label ──────────────────────────────────────────────────────────
    labeler = RuleLabeler(timeframe=timeframe)
    labeled_df = labeler.label_dataframe(df)

    console.print("\nLabel distribution:")
    console.print(labeled_df["label"].value_counts().to_string())

    if dry_run:
        console.print("\n[yellow]Dry run — nothing saved.[/yellow]")
        return

    # ── Save ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pair}_{timeframe}.csv"

    if output_path.exists() and not overwrite:
        console.print(
            f"\n[yellow]Output already exists at {output_path}. "
            f"Use --overwrite to replace.[/yellow]"
        )
        raise typer.Exit(1)

    labeled_df[["label"]].to_csv(output_path)
    console.print(f"\n[green]Labels saved → {output_path}[/green]")
    console.print(
        f"Next step: [bold]guapbot regime fit --pair {pair} --timeframe {timeframe}[/bold]"
    )


if __name__ == "__main__":
    app()
