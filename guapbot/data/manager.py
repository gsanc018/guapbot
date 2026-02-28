"""
guapbot/data/manager.py

DataManager: high-level interface for all price data access.
This is what every other layer calls — never KrakenClient or ParquetStore directly.

Public API:
    manager.fetch(pair, interval)           → DataFrame from cache (CSV-imported data)
    manager.full_history(pair, interval)    → TODO: backfill via GET /public/Trades
    manager.latest_price(pair)              → float (REST ticker)
    manager.available_pairs()              → list of valid pair strings
    manager.cache_info()                    → dict summary of what's cached

Historical data is loaded via KrakenCSVImporter (guapbot/data/importer.py).
Backfilling recent bars (2026 onwards) requires GET /public/Trades — see full_history().
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
        Incremental updates (new bars since the last cached timestamp) are not
        yet implemented — see full_history() for the planned approach.

        TODO: once get_trades() is implemented, add an incremental update step:
            last_ts = self._store.last_timestamp(pair, interval)
            if last_ts is not None:
                new_bars = aggregate_trades(
                    self._client.get_trades(pair, since=last_ts_nanoseconds)
                )
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
        Backfill ALL history for pair/interval using raw tick data.

        The GET /public/OHLC endpoint is limited to ~720 bars and is unreliable
        for full history. Use GET /public/Trades instead, which returns all ticks
        and can be paginated from the beginning to now.

        TODO: implement using KrakenClient.get_trades():
            since = 0
            all_rows = []
            while True:
                rows, last = self._client.get_trades(pair=pair, since=since)
                if not rows:
                    break
                all_rows.extend(rows)
                if str(last) == str(since):   # stalled
                    break
                since = last
            Then resample all_rows → OHLCV time bars for each interval and save
            via self._store.save(). Each tick row from Kraken is:
            [price, volume, time, buy_sell, market_limit, misc, trade_id]
            Use pd.DataFrame(all_rows).resample(interval).agg({...}) to create bars.

        For now, historical data is available via KrakenCSVImporter (CSV files
        cover from coin inception through 2025-12-31).
        """
        raise NotImplementedError(
            "full_history() via trades not yet implemented. "
            "Load historical CSV data with KrakenCSVImporter. "
            "See the docstring for the get_trades() backfill pattern."
        )

    def backfill_trades(self, pair: str, since: int = 0) -> dict[str, pd.DataFrame]:
        """
        Fetch all trades since `since` nanosecond cursor, aggregate to all intervals,
        and append to the store.

        Args:
            pair:  One of XBTUSD, ETHUSD, ETHBTC.
            since: Nanosecond cursor returned by a previous get_trades() call, or 0
                   to start from the earliest trade available on Kraken.
                   To convert from a Unix-second timestamp:
                       since = int(last_bar_ts.timestamp()) * 1_000_000_000

        Returns:
            Dict of {interval: full_cached_DataFrame} for each interval that received
            new bars. Empty dict if no trades were fetched.

        Note:
            Fetching 2 months of BTC ticks ≈ 10M trades ≈ 10,000 API pages.
            Expect roughly 1 hour per pair at Kraken's ~3 req/s public rate limit.
        """
        if pair not in ALL_PAIRS:
            raise ValueError(f"Unknown pair '{pair}'. Valid: {ALL_PAIRS}")

        logger.info("Starting trade backfill for %s (since cursor=%s)", pair, since)

        all_rows: list = []
        cursor = since
        page = 0

        while True:
            rows, last = self._client.get_trades(pair=pair, since=cursor)
            if not rows:
                logger.info("No trades returned — backfill complete for %s", pair)
                break

            all_rows.extend(rows)
            page += 1

            if page % 100 == 0:
                logger.info(
                    "  %s: page %d, %d ticks accumulated (cursor=%s)",
                    pair, page, len(all_rows), cursor,
                )

            if str(last) == str(cursor):
                logger.info("Cursor stalled — backfill complete for %s", pair)
                break

            cursor = last
            time.sleep(0.34)  # ~3 req/s — stay within Kraken public rate limit

        logger.info("Fetched %d total ticks for %s (%d pages)", len(all_rows), pair, page)

        if not all_rows:
            logger.warning("No trades fetched for %s — store unchanged", pair)
            return {}

        ticks = rows_to_tick_df(all_rows)
        results: dict[str, pd.DataFrame] = {}

        for interval in ALL_INTERVALS:
            bars = aggregate_trades_to_ohlcv(ticks, interval)
            if bars.empty:
                logger.warning("No %s bars aggregated for %s", interval, pair)
                continue
            merged = self._store.append(pair, interval, bars)
            results[interval] = merged
            logger.info(
                "  %s %s: +%d new bars appended (total cached: %d)",
                pair, interval, len(bars), len(merged),
            )

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

        TODO: implement once full_history() / get_trades() is ready.
        For now, load historical data with KrakenCSVImporter.
        """
        raise NotImplementedError(
            "download_all() not yet implemented — see full_history() for the plan."
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(pair: str, interval: str) -> None:
        if pair not in ALL_PAIRS:
            raise ValueError(f"Unknown pair '{pair}'. Valid: {ALL_PAIRS}")
        if interval not in ALL_INTERVALS:
            raise ValueError(f"Unknown interval '{interval}'. Valid: {ALL_INTERVALS}")
