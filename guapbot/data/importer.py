"""
guapbot/data/importer.py

Reads Kraken's official OHLCVT CSV files and saves them to the Parquet cache.

Kraken CSV filename format: {PAIR}_{interval_minutes}.csv
e.g. XBTUSD_60.csv, ETHUSD_240.csv, XETHXXBT_1440.csv

Kraken CSV columns (no header):
  timestamp, open, high, low, close, volume, trades

Usage:
    guapbot data import --dir data/raw/master_q4
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from guapbot.data.store import ParquetStore
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Map Kraken CSV pair names → GuapBot pair names
PAIR_MAP = {
    "XBTUSD":   "XBTUSD",
    "XXBTZUSD": "XBTUSD",
    "ETHUSD":   "ETHUSD",
    "XETHZUSD": "ETHUSD",
    "ETHBTC":   "ETHBTC",
    "ETHXBT":   "ETHBTC",    
    "XETHXXBT": "ETHBTC",
}

# Map interval minutes → GuapBot interval string
INTERVAL_MAP = {
    "60":   "1h",
    "240":  "4h",
    "1440": "1d",
}

# Which intervals we actually want to import
WANTED_INTERVALS = set(INTERVAL_MAP.keys())


class KrakenCSVImporter:
    """
    Reads Kraken OHLCVT CSV files from a directory and loads them into
    the Parquet cache.
    """

    def __init__(self, store: Optional[ParquetStore] = None) -> None:
        self._store = store or ParquetStore()

    def import_dir(self, directory: Path) -> dict:
        """
        Scan a directory for Kraken CSV files and import the ones we need.

        Returns a summary dict of what was imported.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all CSV files recursively (handles nested zip extraction)
        csv_files = list(directory.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {directory}")

        results = {}
        for csv_path in sorted(csv_files):
            result = self._try_import(csv_path)
            if result:
                pair, interval, bars = result
                key = f"{pair}_{interval}"
                results[key] = bars
                logger.info(f"  ✓ {key}: {bars:,} bars")

        return results

    def _try_import(self, csv_path: Path) -> Optional[tuple[str, str, int]]:
        """
        Try to import a single CSV file.
        Returns (pair, interval, num_bars) if successful, None if skipped.
        """
        # Parse filename: e.g. "XBTUSD_60.csv" or "XXBTZUSD_60.csv"
        stem = csv_path.stem  # e.g. "XBTUSD_60"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            return None

        raw_pair, interval_min = parts

        # Skip intervals we don't need
        if interval_min not in WANTED_INTERVALS:
            return None

        # Map to GuapBot pair name
        pair = PAIR_MAP.get(raw_pair.upper())
        if pair is None:
            return None

        interval = INTERVAL_MAP[interval_min]

        try:
            df = self._read_csv(csv_path)
            if df.empty:
                logger.warning(f"  Empty file: {csv_path.name}")
                return None

            # Merge with any existing cache (e.g. combining multiple quarterly files)
            existing = self._store.load(pair, interval)
            if not existing.empty:
                merged = pd.concat([existing, df])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                self._store.save(pair, interval, merged)
                bars = len(merged)
            else:
                self._store.save(pair, interval, df)
                bars = len(df)

            return pair, interval, bars

        except Exception as exc:
            logger.error(f"  Failed to import {csv_path.name}: {exc}")
            return None

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """
        Read a Kraken OHLCVT CSV into a normalized DataFrame.

        Kraken format (no header row):
          timestamp, open, high, low, close, volume, trades
        """
        df = pd.read_csv(
            path,
            header=None,
            names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
        )

        # Timestamp is Unix seconds
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)

        return df.sort_index()
