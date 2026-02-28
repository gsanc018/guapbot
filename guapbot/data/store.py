"""
guapbot/data/store.py

Parquet-backed OHLC cache.
Handles all disk I/O for price data. Nothing else reads/writes Parquet files.

Storage layout:
  data/cache/{pair}_{interval}.parquet
  e.g.  data/cache/XBTUSD_1h.parquet
        data/cache/ETHUSD_4h.parquet
        data/cache/ETHBTC_1d.parquet

Design:
  - load()   → full DataFrame from disk, or empty if file doesn't exist
  - save()   → overwrite file with new DataFrame
  - append() → load existing, merge with new bars (drop duplicates), save
  - last_timestamp() → returns the last bar's timestamp (for incremental fetching)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from guapbot.utils.config import settings
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)


class ParquetStore:
    """
    Manages Parquet files for OHLC data.

    Args:
        cache_dir: Path to the cache directory. Defaults to settings.data_cache_dir.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = Path(cache_dir or settings.data_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def load(self, pair: str, interval: str) -> pd.DataFrame:
        """
        Load the cached DataFrame for a pair/interval.
        Returns an empty DataFrame (with correct columns) if no cache exists.
        """
        path = self._path(pair, interval)
        if not path.exists():
            logger.debug(f"Cache miss: {path.name}")
            return self._empty_df()

        df = pd.read_parquet(path)
        # Ensure index is UTC-aware datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        logger.debug(f"Loaded {len(df)} bars from {path.name}")
        return df.sort_index()

    def save(self, pair: str, interval: str, df: pd.DataFrame) -> None:
        """Overwrite the cache file with df."""
        if df.empty:
            logger.warning(f"Attempted to save empty DataFrame for {pair} {interval} — skipped")
            return
        df = self._validate_ohlcv(df, pair, interval)
        if df.empty:
            logger.warning(f"All bars failed validation for {pair} {interval} — nothing saved")
            return
        path = self._path(pair, interval)
        df.sort_index().to_parquet(path, compression="snappy")
        logger.debug(f"Saved {len(df)} bars to {path.name}")

    def append(self, pair: str, interval: str, new_bars: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new_bars into the existing cache.

        Deduplicates on timestamp index (new bars overwrite old on conflict).
        Returns the full merged DataFrame.
        """
        if new_bars.empty:
            return self.load(pair, interval)

        existing = self.load(pair, interval)
        if existing.empty:
            merged = new_bars
        else:
            merged = pd.concat([existing, new_bars])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()

        self.save(pair, interval, merged)
        added = len(merged) - len(existing)
        logger.info(f"Cache updated {pair} {interval}: +{added} bars → {len(merged)} total")
        return merged

    def last_timestamp(self, pair: str, interval: str) -> Optional[pd.Timestamp]:
        """
        Return the timestamp of the last cached bar, or None if cache is empty.
        Used by DataManager to determine the 'since' parameter for incremental fetches.
        """
        df = self.load(pair, interval)
        if df.empty:
            return None
        return df.index[-1]

    def cache_info(self) -> dict:
        """Return a summary dict of what's cached."""
        info: dict = {}
        for path in sorted(self._cache_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(path, columns=["close"])  # minimal read
                name = path.stem  # e.g. "XBTUSD_1h"
                info[name] = {
                    "bars": len(df),
                    "first": str(df.index[0]) if len(df) else "—",
                    "last": str(df.index[-1]) if len(df) else "—",
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            except Exception as exc:
                info[path.stem] = {"error": str(exc)}
        return info

    def clear(self, pair: str, interval: str) -> None:
        """Delete a cache file (use only in tests / manual resets)."""
        path = self._path(pair, interval)
        if path.exists():
            path.unlink()
            logger.info(f"Cleared cache: {path.name}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_ohlcv(self, df: pd.DataFrame, pair: str, interval: str) -> pd.DataFrame:
        """
        Clean and validate an OHLCV DataFrame before writing to disk.

        Checks performed (bad bars are dropped with a warning, never raised):
            1. Index is UTC-aware DatetimeIndex
            2. Duplicate timestamps — keep last
            3. Non-monotonic index (backwards time) — sort and warn
            4. Positive prices — open, high, low, close all > 0
            5. OHLC sanity — high >= low, high >= open/close, low <= open/close
            6. Zero-volume bars — dropped (exchange maintenance / bad data)
            7. NaN in any OHLCV column — dropped

        Returns the cleaned DataFrame (may be shorter than input).
        """
        tag = f"{pair} {interval}"
        original_len = len(df)

        # 1. Ensure UTC-aware DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # 2. Deduplicate timestamps (keep last, same as append)
        dupes = df.index.duplicated().sum()
        if dupes:
            logger.warning(f"Dropped {dupes} duplicate timestamps in {tag}")
            df = df[~df.index.duplicated(keep="last")]

        # 3. Non-monotonic index
        if not df.index.is_monotonic_increasing:
            logger.warning(f"Non-monotonic timestamps in {tag} — sorting")
            df = df.sort_index()

        # 4 & 5. Price sanity
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        if price_cols:
            positive = (df[price_cols] > 0).all(axis=1)
            ohlc_ok = pd.Series(True, index=df.index)
            if {"high", "low"}.issubset(df.columns):
                ohlc_ok &= df["high"] >= df["low"]
            if {"high", "close"}.issubset(df.columns):
                ohlc_ok &= df["high"] >= df["close"]
            if {"low", "close"}.issubset(df.columns):
                ohlc_ok &= df["low"] <= df["close"]
            if {"high", "open"}.issubset(df.columns):
                ohlc_ok &= df["high"] >= df["open"]
            if {"low", "open"}.issubset(df.columns):
                ohlc_ok &= df["low"] <= df["open"]

            bad_price = ~(positive & ohlc_ok)
            if bad_price.any():
                logger.warning(f"Dropped {bad_price.sum()} bars with bad OHLC values in {tag}")
                df = df[~bad_price]

        # 6. Zero-volume bars
        if "volume" in df.columns:
            zero_vol = df["volume"] <= 0
            if zero_vol.any():
                logger.warning(f"Dropped {zero_vol.sum()} zero-volume bars in {tag}")
                df = df[~zero_vol]

        # 7. NaN in any OHLCV column
        ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        if ohlcv_cols:
            nan_mask = df[ohlcv_cols].isna().any(axis=1)
            if nan_mask.any():
                logger.warning(f"Dropped {nan_mask.sum()} bars with NaN OHLCV values in {tag}")
                df = df[~nan_mask]

        dropped = original_len - len(df)
        if dropped:
            logger.info(f"Validation: removed {dropped} bad bars from {tag} ({len(df)} remain)")

        return df

    def _path(self, pair: str, interval: str) -> Path:
        return self._cache_dir / f"{pair}_{interval}.parquet"

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "trades"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
