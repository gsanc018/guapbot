"""
guapbot/data/manager.py

DataManager: high-level interface for all price data access.
This is what every other layer calls — never KrakenClient or ParquetStore directly.

Public API:
    manager.fetch(pair, interval)              → DataFrame from cache (CSV-imported data)
    manager.full_history(pair, interval)       → backfill all history via GET /public/Trades
    manager.backfill_trades(pair, since)       → incremental backfill from a nanosecond cursor
    manager.latest_price(pair)                 → float (REST ticker)
    manager.available_pairs()                  → list of valid pair strings
    manager.cache_info()                       → dict summary of what's cached

Historical data is loaded via KrakenCSVImporter (guapbot/data/importer.py).
Backfilling recent bars (2026 onwards) uses backfill_trades() / full_history().
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from guapbot.data.aggregator import aggregate_trades_to_ohlcv, rows_to_tick_df
from guapbot.data.kraken_client import KrakenClient
from guapbot.data.store import ParquetStore
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# All pairs and intervals GuapBot uses
ALL_PAIRS = ["XBTUSD", "ETHUSD", "ETHBTC"]
ALL_INTERVALS = ["1h", "4h", "1d"]

# Flush to store every this many pages during backfill (~200k ticks, ~28h of BTC trades).
# Keeps peak memory to ~100 MB instead of accumulating all ticks at once.
_BACKFILL_CHUNK_PAGES = 200

# Hours of ticks to carry forward across flushes.
# Must be > max interval (1d = 24h) to avoid splitting daily bars at a flush boundary.
_BACKFILL_BAR_BUFFER_H = 25


class DataManager:
    """
    Orchestrates data fetching, caching, and serving for all layers.

    Args:
        client:    KrakenClient instance (injected for testability)
        store:     ParquetStore instance (injected for testability)
    """

    def __init__(
        self,
        client: Optional[KrakenClient] = None,
        store: Optional[ParquetStore] = None,
    ) -> None:
        self._client = client or KrakenClient()
        self._store = store or ParquetStore()

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def fetch(self, pair: str, interval: str = "1h") -> pd.DataFrame:
        """
        Return the cached DataFrame for pair/interval.

        Historical data is loaded into the cache via KrakenCSVImporter.
        For recent bars not yet in the CSV, call backfill_trades() first.

        TODO: wire incremental updates when BarBuilder is implemented (Session 7+):
            last_ts = self._store.last_timestamp(pair, interval)
            if last_ts is not None:
                since_ns = int(last_ts.timestamp()) * 1_000_000_000
                rows, _ = self._client.get_trades(pair, since=since_ns)
                new_bars = aggregate_trades_to_ohlcv(rows_to_tick_df(rows), interval)
                return self._store.append(pair, interval, new_bars)
        """
        self._validate(pair, interval)
        cached = self._store.load(pair, interval)
        if cached.empty:
            logger.warning(
                "No cached data for %s %s — import CSV first via KrakenCSVImporter", pair, interval
            )
        return cached

    def full_history(self, pair: str, interval: str = "1h") -> pd.DataFrame:
        """
        Download full tick history for pair from the beginning of Kraken's records,
        aggregate to the requested interval, and return the cached DataFrame.

        This is a wrapper around backfill_trades(since=0). Expects to run for
        several hours per pair (BTC full history ≈ 10M+ trades).
        """
        self._validate(pair, interval)
        results = self.backfill_trades(pair, since=0)
        return results.get(interval, pd.DataFrame())

    def backfill_trades(self, pair: str, since: Optional[int] = None) -> dict[str, pd.DataFrame]:
        """
        Fetch all trades since `since` nanosecond cursor, aggregate to all intervals,
        and append to the store.

        Memory-safe: flushes completed bars to disk every _BACKFILL_CHUNK_PAGES pages
        (~200k ticks) and retains only the last _BACKFILL_BAR_BUFFER_H hours of ticks
        in memory to protect against incomplete bars at flush boundaries.

        Args:
            pair:  One of XBTUSD, ETHUSD, ETHBTC.
            since: Nanosecond cursor. Defaults to None — auto-resumes from the
                   last cached 1h bar's timestamp. Pass 0 to start from the very
                   beginning of Kraken's trade history (slow — full BTC history
                   ≈ 10,000+ API pages ≈ 1 hour).
                   To set a specific start point:
                       since = int(unix_ts_seconds * 1_000_000_000)

        Returns:
            Dict of {interval: full_cached_DataFrame} for each interval that received
            new bars. Empty dict if no trades were fetched.

        Note:
            Fetching 2 months of BTC ticks ≈ 10M trades ≈ 10,000 API pages.
            Expect roughly 1 hour per pair at Kraken's ~3 req/s public rate limit.
        """
        if pair not in ALL_PAIRS:
            raise ValueError(f"Unknown pair '{pair}'. Valid: {ALL_PAIRS}")

        if since is None:
            last_ts = self._store.last_timestamp(pair, "1h")
            if last_ts is not None:
                since = int(last_ts.timestamp()) * 1_000_000_000
                logger.info("Resuming %s from last cached bar: %s (cursor=%s)", pair, last_ts, since)
            else:
                since = 0
                logger.info("No cache for %s — fetching from beginning of Kraken history", pair)

        logger.info("Starting trade backfill for %s (since cursor=%s)", pair, since)

        pending: list = []   # ticks not yet flushed to disk
        cursor = since
        page = 0
        total_ticks = 0
        results: dict[str, pd.DataFrame] = {}

        def _flush(is_final: bool) -> None:
            nonlocal pending
            if not pending:
                return

            ticks = rows_to_tick_df(pending)
            last_ts = ticks["timestamp"].max()

            for interval in ALL_INTERVALS:
                bars = aggregate_trades_to_ohlcv(ticks, interval)
                if bars.empty:
                    continue

                if not is_final:
                    # Only save bars that closed > _BACKFILL_BAR_BUFFER_H hours before
                    # the last tick, so we never save a bar that still has incoming ticks.
                    safe_cutoff = last_ts - pd.Timedelta(hours=_BACKFILL_BAR_BUFFER_H)
                    bars = bars[bars.index < safe_cutoff]

                if bars.empty:
                    continue

                merged = self._store.append(pair, interval, bars)
                results[interval] = merged

            if not is_final:
                # Retain the most recent _BACKFILL_BAR_BUFFER_H hours of ticks so the
                # next chunk can complete any bar that straddles the flush boundary.
                cutoff_unix = (last_ts - pd.Timedelta(hours=_BACKFILL_BAR_BUFFER_H)).timestamp()
                pending = [r for r in pending if float(r[2]) >= cutoff_unix]

        while True:
            rows, last = self._client.get_trades(pair=pair, since=cursor)
            if not rows:
                logger.info("No trades returned — backfill complete for %s", pair)
                break

            pending.extend(rows)
            total_ticks += len(rows)
            page += 1

            if page % _BACKFILL_CHUNK_PAGES == 0:
                _flush(is_final=False)
                logger.info(
                    "  %s: page %d, %d total ticks, ~%d ticks in buffer",
                    pair, page, total_ticks, len(pending),
                )

            if str(last) == str(cursor):
                logger.info("Cursor stalled — backfill complete for %s", pair)
                break

            cursor = last
            time.sleep(0.34)  # ~3 req/s — stay within Kraken public rate limit

        _flush(is_final=True)

        if not results:
            logger.warning("No trades fetched for %s — store unchanged", pair)
            return {}

        logger.info("Backfill complete for %s: %d ticks, %d pages", pair, total_ticks, page)
        for interval, df in results.items():
            logger.info("  %s %s: %d total bars cached", pair, interval, len(df))

        return results

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def latest_price(self, pair: str) -> float:
        """Return the latest price for a pair via REST ticker."""
        ticker = self._client.get_ticker(pair)
        return ticker["last"]

    def available_pairs(self) -> list[str]:
        return ALL_PAIRS.copy()

    def cache_info(self) -> dict:
        return self._store.cache_info()

    # ------------------------------------------------------------------
    # Batch download (used by CLI)
    # ------------------------------------------------------------------

    def download_all(self) -> None:
        """
        Download full history for all pairs and all intervals.
        Called by: guapbot data download
        """
        logger.info("Starting full data download for all pairs and intervals...")
        total = len(ALL_PAIRS) * len(ALL_INTERVALS)
        count = 0
        for pair in ALL_PAIRS:
            count += 1
            logger.info("[%d/%d] %s", count, total, pair)
            try:
                results = self.backfill_trades(pair, since=0)
                for interval, df in results.items():
                    logger.info("  %s %s: %d bars", pair, interval, len(df))
            except Exception as exc:
                logger.error("  %s failed: %s", pair, exc)
        logger.info("Download complete.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(pair: str, interval: str) -> None:
        if pair not in ALL_PAIRS:
            raise ValueError(f"Unknown pair '{pair}'. Valid: {ALL_PAIRS}")
        if interval not in ALL_INTERVALS:
            raise ValueError(f"Unknown interval '{interval}'. Valid: {ALL_INTERVALS}")
