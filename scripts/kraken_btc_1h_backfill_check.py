#!/usr/bin/env python3
"""
Manual Kraken backfill sanity check for BTC/USD 1h candles.

This script is intentionally verbose:
1. Loads archived Kraken CSV history (if available).
2. Uses a loop to walk backward by changing `since` each iteration.
3. Merges archive + API bars into one canonical dataset.
4. Builds the dataset from 2025-01-01 UTC to now.
5. Prints the full dataset.
6. Prints FAILED if row count is below a threshold (default: 8000).

Usage:
    .venv/bin/python scripts/kraken_btc_1h_backfill_check.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.kraken.com/0/public/OHLC"
PAIR = "XXBTZUSD"  # BTC/USD on Kraken
INTERVAL_MIN = 60  # 1h candles


def fetch_ohlc(since: int | None) -> tuple[pd.DataFrame, int]:
    """Fetch one OHLC page from Kraken and return (df, last_cursor)."""
    params: dict[str, int | str] = {"pair": PAIR, "interval": INTERVAL_MIN}
    if since is not None:
        params["since"] = int(since)

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error"):
        raise RuntimeError(f"Kraken error: {payload['error']}")

    result = payload["result"]
    pair_key = next(k for k in result.keys() if k != "last")
    rows = result[pair_key]
    last_cursor = int(result["last"])

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    for col in ["open", "high", "low", "close", "vwap", "volume", "trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, last_cursor


def load_csv_archive(path: Path) -> pd.DataFrame:
    """Load Kraken OHLC CSV archive if present."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df["vwap"] = pd.NA  # CSV export has no VWAP column
    for col in ["open", "high", "low", "close", "volume", "trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "vwap", "volume", "trades"]]


def fetch_api_backwards(max_pages: int, sleep_s: float) -> pd.DataFrame:
    """
    Walk backwards using a `since` loop and append pages.

    Kraken 1h OHLC typically returns only a recent window (~720 bars). This loop
    still follows the requested strategy: change `since` each iteration based on
    the oldest timestamp we got, attempt an older page, and stop when it stalls.
    """
    frames: list[pd.DataFrame] = []

    since = 0  # first page
    seen_since: set[int] = set()
    prev_oldest: pd.Timestamp | None = None
    step_seconds = 720 * 3600

    for page in range(1, max_pages + 1):
        try:
            page_df, last_cursor = fetch_ohlc(since=since)
        except Exception as exc:  # noqa: BLE001
            print(f"Page {page:03d}: API error ({exc}), stopping API loop")
            break

        if page_df.empty:
            print(f"Page {page:03d}: empty page, stopping")
            break

        page_first = page_df.index.min()
        page_last = page_df.index.max()
        print(
            f"Page {page:03d}: bars={len(page_df):4d}  "
            f"{page_first} -> {page_last}  cursor={last_cursor}"
        )
        frames.append(page_df)

        oldest = page_first
        if prev_oldest is not None and oldest >= prev_oldest:
            print("Oldest bar did not move backward, stopping API loop")
            break
        prev_oldest = oldest

        next_since = int((oldest - pd.Timedelta(seconds=step_seconds)).timestamp())
        if next_since <= 0:
            print("Reached non-positive since, stopping API loop")
            break
        if next_since in seen_since:
            print("since value repeated, stopping API loop")
            break

        seen_since.add(next_since)
        since = next_since
        time.sleep(sleep_s)

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames)
    return full[~full.index.duplicated(keep="last")].sort_index()


def build_dataset(
    start_utc: pd.Timestamp,
    csv_path: Path,
    max_pages: int,
    sleep_s: float,
) -> tuple[pd.DataFrame, int, int]:
    """Merge archive + API bars and return final dataset + source counts."""
    archive = load_csv_archive(csv_path)
    api = fetch_api_backwards(max_pages=max_pages, sleep_s=sleep_s)

    archive_rows = len(archive)
    api_rows = len(api)

    if archive.empty and api.empty:
        raise RuntimeError("No data available from CSV archive or Kraken API")

    merged = (
        pd.concat([archive, api])
        if not archive.empty and not api.empty
        else (archive if not archive.empty else api)
    )
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    now_utc = pd.Timestamp(datetime.now(UTC))
    merged = merged[(merged.index >= start_utc) & (merged.index <= now_utc)]
    return merged, archive_rows, api_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build BTC/USD 1h dataset from 2025-01-01 UTC to now using "
            "Kraken OHLC cursor pagination."
        )
    )
    parser.add_argument("--start", default="2025-01-01T00:00:00Z", help="UTC start timestamp")
    parser.add_argument("--min-rows", type=int, default=8000, help="FAIL threshold")
    parser.add_argument("--max-pages", type=int, default=300, help="Pagination safety cap")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between API calls (sec)")
    parser.add_argument(
        "--csv",
        default="data/raw/XBTUSD_60.csv",
        help="Path to Kraken archive CSV (timestamp,open,high,low,close,volume,trades)",
    )
    args = parser.parse_args()

    start_utc = pd.Timestamp(args.start)
    if start_utc.tz is None:
        start_utc = start_utc.tz_localize("UTC")
    else:
        start_utc = start_utc.tz_convert("UTC")

    print(f"Building BTC 1h dataset from {start_utc} to now (UTC)")
    print(f"Using pair={PAIR}, interval={INTERVAL_MIN}m, min_rows={args.min_rows}")

    df, archive_rows, api_rows = build_dataset(
        start_utc=start_utc,
        csv_path=Path(args.csv),
        max_pages=args.max_pages,
        sleep_s=args.sleep,
    )
    print(f"Source rows: archive={archive_rows:,}, api_window={api_rows:,}")

    # Print full dataset (not truncated)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print("\nFull dataset:")
    print(df.to_string())

    print("\nSummary:")
    print(f"Rows:  {len(df):,}")
    print(f"First: {df.index.min() if not df.empty else 'EMPTY'}")
    print(f"Last:  {df.index.max() if not df.empty else 'EMPTY'}")

    if len(df) < args.min_rows:
        print(f"FAILED: dataset has {len(df):,} rows (< {args.min_rows:,})")
        return 1

    print(f"PASSED: dataset has {len(df):,} rows (>= {args.min_rows:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
